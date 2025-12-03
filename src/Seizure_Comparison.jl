#using Pkg
#Pkg.add("PyCall"); Pkg.add("DSP"); Pkg.add("Statistics"); Pkg.add("Plots"); Pkg.add("FFTW")
using PyCall
using DSP
using Statistics
using Plots
using FFTW

gr() 

#edf loader via mne
function load_edf_channels(path::AbstractString)
    mne = pyimport("mne")
    raw = mne.io.read_raw_edf(path, preload=true)

    chans = Dict{String, Tuple{Vector{Float64}, Float64}}()
    for ch_name in raw.ch_names
        sig = Array{Float64}(raw.get_data(picks=[ch_name])[1, :])
        fs = float(raw.info["sfreq"])
        chans[ch_name] = (sig, fs)
    end
    return chans
end

#simple design: notch around f0 with quality factor Q
function design_notch(fs::Real, f0::Real=60.0, q::Real=30.0, order::Int=2)
    nyq = fs/2
    bw = f0 / q              
    low = (f0 - bw/2)/nyq
    high = (f0 + bw/2)/nyq
    return digitalfilter(Bandstop(low, high), Butterworth(order))
end

#design normalized bandpass between low_hz and high_hz
function design_bandpass(fs::Real, low_hz::Real=0.5, high_hz::Real=70.0, order::Int=4)
    nyq = fs / 2
    return digitalfilter(Bandpass(low_hz/nyq, high_hz/nyq), Butterworth(order))
end

#apply chain of filters using filtfilt (zero-phase)
function apply_filters(x::AbstractVector{<:Real}, fs::Real; notch_filter=nothing, bp_filter=nothing)
    y = copy(x)
    if notch_filter !== nothing
        y = filtfilt(notch_filter, y)
    end
    if bp_filter !== nothing
        y = filtfilt(bp_filter, y)
    end
    return y
end

line_length(x) = sum(abs.(diff(x))) / max(1, length(x))
rmsval(x) = sqrt(mean(abs2, x))
variance(x) = var(x)

function band_power(x, fs, low, high)
    bp_filter = design_bandpass(fs, low, high)
    y = filtfilt(bp_filter, x)
    return mean(abs2, y)
end

#compute power spectral density
function compute_psd(signal::AbstractVector{<:Real}, fs::Real; max_freq::Real=50.0)
    n = length(signal)
    fft_result = fft(signal)
    power = abs2.(fft_result) ./ n
    freqs = fftfreq(n, fs)
    
    #only positive frequencies
    pos_idx = freqs .>= 0
    freqs = freqs[pos_idx]
    power = power[pos_idx]
    
    #limit to max_freq
    freq_idx = freqs .<= max_freq
    return freqs[freq_idx], power[freq_idx]
end

#extract band powers for standard EEG bands
function extract_band_powers(signal::AbstractVector{<:Real}, fs::Real)

    #dictionary that holds band power frequencies
    bands = Dict(
        "Delta (0.5-4 Hz)" => (0.5, 4.0),
        "Theta (4-8 Hz)" => (4.0, 8.0),
        "Alpha (8-12 Hz)" => (8.0, 12.0),
        "Beta (12-30 Hz)" => (12.0, 30.0),
        "Gamma (30-100 Hz)" => (30.0, 100.0)
    )
    
    powers = Dict{String, Float64}()
    for (name, (low, high)) in bands
        powers[name] = band_power(signal, fs, low, high)
    end
    return powers
end

#keep plot displayed until Enter is pressed
function wait_for_enter(message="Press Enter to continue...")
    println(message)
    readline()
end

#compare two EEG signals
function compare_signals(seizure_path::String, normal_path::String)
    
    #load both files
    println("Loading seizure file: $seizure_path")
    sz_chans = load_edf_channels(seizure_path)
    println("Loading normal file: $normal_path")
    norm_chans = load_edf_channels(normal_path)
    
    #select first available channel
    channel = first(keys(sz_chans))
    println("Using channel: $channel")
    
    if !haskey(sz_chans, channel) || !haskey(norm_chans, channel)
        error("Channel $channel not found in both files")
    end
    
    sz_signal, sz_fs = sz_chans[channel]
    norm_signal, norm_fs = norm_chans[channel]
    
    fs = sz_fs
    
    #take segment of 300 seconds to show EEG signal
    segment_duration = 300.0
    max_samples = Int(min(segment_duration * fs, minimum([length(sz_signal), length(norm_signal)])))
    sz_seg = sz_signal[1:max_samples]
    norm_seg = norm_signal[1:max_samples]
    
    #preprocess both signals using updated filter functions
    notch_filt = design_notch(fs, 60.0, 30.0, 2)
    bp_filt = design_bandpass(fs, 0.5, min(50.0, fs/2 - 1.0), 4)
    
    sz_filt = apply_filters(sz_seg, fs; notch_filter=notch_filt, bp_filter=bp_filt)
    norm_filt = apply_filters(norm_seg, fs; notch_filter=notch_filt, bp_filter=bp_filt)
    
    #compute statistics (range, standard deviation, mean)
    println("\n-- AMPLITUDE COMPARISON --")
    sz_amp = Dict(
        "RMS" => rmsval(sz_filt),
        "Peak-to-Peak" => maximum(sz_filt) - minimum(sz_filt),
        "Std Dev" => std(sz_filt),
        "Mean Abs" => mean(abs.(sz_filt))
    )
    norm_amp = Dict(
        "RMS" => rmsval(norm_filt),
        "Peak-to-Peak" => maximum(norm_filt) - minimum(norm_filt),
        "Std Dev" => std(norm_filt),
        "Mean Abs" => mean(abs.(norm_filt))
    )
    
    println("\nSeizure Amplitudes:")
    for (k, v) in sz_amp
        println("  $k: $(round(v, digits=4))")
    end
    println("\nNormal Amplitudes:")
    for (k, v) in norm_amp
        println("  $k: $(round(v, digits=4))")
    end
    println("\nRatio (Seizure/Normal):")
    for k in keys(sz_amp)
        ratio = sz_amp[k] / norm_amp[k]
        println("  $k: $(round(ratio, digits=2))x")
    end
    
    #compute each band power
    println("\n-- FREQUENCY BAND POWER COMPARISON --")
    sz_bands = extract_band_powers(sz_filt, fs)
    norm_bands = extract_band_powers(norm_filt, fs)
    
    println("\nSeizure Band Powers:")
    for (band, power) in sort(collect(sz_bands))
        println("  $band: $(round(power, digits=6))")
    end
    println("\nNormal Band Powers:")
    for (band, power) in sort(collect(norm_bands))
        println("  $band: $(round(power, digits=6))")
    end
    println("\nRatio (Seizure/Normal):")
    for band in sort(collect(keys(sz_bands)))
        ratio = sz_bands[band] / norm_bands[band]
        println("  $band: $(round(ratio, digits=2))x")
    end
    
    #compute PSDs
    sz_freqs, sz_psd = compute_psd(sz_filt, fs; max_freq=50.0)
    norm_freqs, norm_psd = compute_psd(norm_filt, fs; max_freq=50.0)
    
    #create comparison plots (only time domain and PSD)
    t_sz = (0:length(sz_filt)-1) ./ fs
    t_norm = (0:length(norm_filt)-1) ./ fs
    
    #plot 1: Time domain comparison
    p1 = plot(t_sz, sz_filt, label="Seizure", color=:red, alpha=0.7, linewidth=1)
    plot!(p1, t_norm, norm_filt, label="Normal", color=:blue, alpha=0.7, linewidth=1)
    xlabel!(p1, "Time (s)")
    ylabel!(p1, "Amplitude (μV)")
    title!(p1, "Time Domain Comparison - Channel: $channel")
    
    #plot 2: Power Spectral Density
    p2 = plot(sz_freqs, 10*log10.(sz_psd .+ eps()), label="Seizure", color=:red, linewidth=2)
    plot!(p2, norm_freqs, 10*log10.(norm_psd .+ eps()), label="Normal", color=:blue, linewidth=2)
    xlabel!(p2, "Frequency (Hz)")
    ylabel!(p2, "Power (dB)")
    title!(p2, "Power Spectral Density Comparison")
    
    #combine plots
    final_plot = plot(p1, p2, layout=(2,1), size=(1200, 800), margin=5Plots.mm)
    display(final_plot)
    
    wait_for_enter("Press Enter to exit:")
    
    return (seizure=(signal=sz_filt, fs=fs, bands=sz_bands, amps=sz_amp, psd=(sz_freqs, sz_psd)),
            normal=(signal=norm_filt, fs=fs, bands=norm_bands, amps=norm_amp, psd=(norm_freqs, norm_psd)))
end

#main function for command line usage
function run_comparison()
    println("-- EEG Seizure vs Normal Comparison --\n")
    
    #get seizure file path
    print("Enter seizure file path: ")
    seizure_path = String(strip(readline()))
    
    #get normal awake file path
    print("Enter normal awake file path: ")
    normal_path = String(strip(readline()))
    
    results = compare_signals(seizure_path, normal_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end