
using PlutoUI
using FFTW, Statistics, LinearAlgebra

n = 1000
x = range(0, 1, length=n)
ς = Slider(0.5:0.5:2.0)  # noise level

μ = 3*x.*sin.(2pi*x) # periodic signal in time domain
#μ = 6*sqrt.(abs.(x .- 0.5)) # this one is difficult to estimate

# Model: Signal distorted by white noise
y = μ + ς*randn(n)  

# Prior power decay of prior samples in frequency domain
α = Slider(0.5:0.5:2.0)
# α = 3.0 for type integrated Brownian motion
# α = 2.0 for type Brownian motion prior
# α = 1.0 for a very rough prior (pink noise)

H = (α - 1)/2 # equivalent Hurst exponent


# Fourier matrix
# ℱ = [exp(2pi*im*i*j/n) for i in fftshift(-n÷2:n÷2-1), j in fftshift(-n÷2:n÷2-1)]

# Define prior on unit interval in frequency domain
# by choosing power decay
function freq_decay(n, α)  #ifftshift(-n/2:n/2-1)
    fr = 1 ./ fftfreq(n)
    fr[1] = 2n # allow for intercept
    fr = abs.(fr) .^(α/2) # scale frequencies to "unit 
    fr/norm(fr)*sqrt(n)
end  

# Canonical posterior contraction rate
contract(n, H) =  n^(-H/(1 + 2H))


# Compute posterior mean
γ = ς^(-2) .+ (freq_decay(n, α)).^(-2) # posterior precision in frequency domain
μ̂ = real(ifft(γ.\(fft(ς^(-2)*y))))
m = μ̂ + sqrt(n)*real(ifft(sqrt.(γ).\randn(Complex{Float64},n))); 
# note: `fft(x)/sqrt(length(x))` is isometric


# Compute L2 estimation error, compare with theoretical size
@show norm(μ̂ - μ)/sqrt(n),  ς*contract(n, H)

# We can even compute the posterior covariance 



# Use circulant property of covariance in time domain, could be done with https://github.com/JuliaMatrices/ToeplitzMatrices.jl
σ = 1.96*sqrt.(abs.((ifft(diag(inv(Diagonal(γ)))))))[1]

# Equivalent to
# Γ = ℱ*Diagonal(freq_decay(n, α).^(-2))*ℱ'/n/n # ≈ ifft(Diagonal(n. + freq_decay(n, α).^(-2))) 
# σ = 1.96sqrt.(real(diag(inv((I + Γ)))))

#=
# Plot
using Plots
p = scatter(x, y, markersize=0.5, label="obs")
plot!(x, μ̂, color=:blue, ribbon=σ,fillalpha=0.2, label="posterior mean")
plot!(x, μ, color=:green, label="truth")
z = sqrt(n)*real(ifft(randn(Complex{Float64},n).*abs.(freq_decay(n, α)))); 
plot!(x, z, label="prior sample")
plot!(x, m, label="posterior sample")

p
=#