using DrWatson
@quickactivate "MVMSTimePred"

using MVMSTimePred
using Flux
using Flux: @epochs, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse
using CUDA; CUDA.allowscalar(false)
using Lazy: @>, @>>, @as
using BSON
using Statistics
using IterTools: ncycle

args = length(ARGS) == 0 ? ["-c"] : ARGS #Default is CPU
dev = args[1] == "-g" ? gpu : cpu #Device

println("Loading data")
data = BSON.load(datadir("lorenzX.bson"))[:X]
X, Y = create_xy(data, 7, 4)
X = @> X begin
  eachslice(dims = 3)
  Flux.unstack.(2)
end
Y = Flux.unstack(Y, 3)

Xtrain = X[1:150] .|> dev
Xtest = X[151:end] .|> dev
Ytrain = Y[1:150] .|> dev
Ytest = Y[151:end] .|> dev

train_data = DataLoader((Xtrain, Ytrain); batchsize = 16)
println("Training and testing datasets created")

m = LSTnetCell(3, 32, 4)
println("LSTnet created")

loss(x, y) = mean(Flux.mse.(m.(x), y))

opt = Flux.ADAM()

ps = Flux.params(m)

evalcb() = @show(loss(Xtest, Ytest))
println("Current testing loss is", evalcb())

println("Begin training")
Flux.train!(loss, ps, ncycle(train_data, 100), opt; cb = throttle(evalcb, 5))
