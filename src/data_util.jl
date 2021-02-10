"""
  create_xy(data, lagx, leady)

The input timeseries data is of form `nₓ×nₜ`. Creates output with dimension `nₓ⨱nₜₗ⨱nₑ`.
"""
function create_xy(data, lagx, leady)
    leads = lagx + leady #Create data for total timesteps of lead
    lead_data = zeros(size(data, 1), leads, size(data, 2))
    for i = 1:leads
        circshift!(view(lead_data, :, i, :), data, (0, 1-i))
        #circshift creates a lag for positive `i`. I am creating a lead by
        #making i negative. 1 is added to account for zero lead
    end
    lead_data = lead_data[:, :, 1:(size(data, 2) - leads + 1)]
    return lead_data[:, 1:lagx, :], lead_data[:, (lagx+1):(lagx+leady), :]
end
