module MVMSTimePred

using Flux
using Flux: @functor
import Base: show

include("data_util.jl")

export create_xy,
       Repeater,
       LSTnetCell

"""
  Repeater(α::AbstractArray)
  Repeater(sz::Integer...)

Creates a layer that allows replicates an input layer to multiple horizon. Unstacking is always done on last dimension
"""
struct Repeater{A<:AbstractArray, D<:Integer}
  α::A
  dims::D
end

Repeater(sz::Integer...; initα = i -> ones(Float32, i)) = Repeater(initα(sz))
Repeater(A::AbstractArray) = Repeater(A, ndims(A))

@functor Repeater

(r::Repeater)(x) = Flux.unstack(r.α .* x, r.dims)

show(io::IO, r::Repeater) = print(io, "Repeater(", size(r.α), ")")

struct LSTnetCell{A, B, C, D}
  Encoder::A
  DenseLayer::B
  RepeatVec::C
  Decoder::D
end

"""
  LSTnetCell(feature_length, hidden, horizon)

Creates a LSTnet layer with LSTM encoder. A hidden fully connected dense layer.
The horizon is the number of future timesteps to be predicted
"""
function LSTnetCell(feature_length::Integer, hidden::Integer, horizon::Integer)
  EN = LSTM(feature_length, hidden)
  HID = Dense(hidden, hidden, relu)
  RE = Repeater(hidden, horizon)
  DE = Chain(LSTM(hidden, feature_length),
    Dense(feature_length, feature_length))
  return LSTnetCell(EN, HID, RE, DE)
end

function show(io::IO, m::LSTnetCell)
  feat_len = size(m.Encoder.cell.Wi,2)
  hid_len = size(m.Encoder, 1) ÷ 4
  horizon = size(m.RepeatVec.α, 2)
  print(io, "Encoder(LSTM(", feat_len, ",", hid_len, ")) -> ")
  print(io, "Fully Connected(Dense(", hid_len, ",", hid_len, ", relu)) -> ")
  print(io, "Repeater(Ones(", hid_len, ",", horizon, ")) -> ")
  print(io, "Decoder(LSTM(", hid_len, ",", feat_len, "))")
end

function (m::LSTnetCell)(x)
  Flux.reset!(m)
  # @info "Network reset"
  # @info "Input data unstacked $(size(Xbatch))"
  for i ∈ x
    m.Encoder(i)
  end
  X1 = m.Encoder.state[1]
  # @info "Context vector of $(size(X1))"
  X2 = m.DenseLayer(X1)
  # @info "Nonlinear context vector of $(size(X2))"
  X3 = m.RepeatVec(X2)
  X4 = mapreduce(m.Decoder, hcat, X3)
  # @info "Final output vector of $(size(X3))"
  return X4
end

Flux.@functor LSTnetCell
Flux.params(m::LSTnetCell) = Flux.params(m.Encoder, m.DenseLayer, m.Decoder)
Flux.reset!(m::LSTnetCell) = Flux.reset!.((m.Encoder, m.Decoder))

end #Module
