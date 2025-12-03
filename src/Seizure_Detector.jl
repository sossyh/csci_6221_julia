#using Pkg
#Pkg.add("PyCall"); Pkg.add("DSP"); Pkg.add("Statistics"); Pkg.add("Plots")
using PyCall
using DSP
using Statistics
using Plots

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

#filters
@inline dist2(a,b) = error("unused")  #placeholder

function bandpass_filter(x::AbstractVector{<:Real}, fs::Real, low::Real, high::Real; order::Int=4)
    ny = fs/2
    lo = max(low/ny, 1e-8); hi = min(high/ny, 0.999999)
    sos = DSP.Filters.digitalfilter(DSP.Filters.Bandpass(lo, hi), DSP.Filters.Butterworth(order))
    return DSP.filtfilt(sos, collect(Float64, x))
end

function notch_filter(x::AbstractVector{<:Real}, fs::Real; f0::Real=50.0, bw::Real=2.0, order::Int=2)
    ny = fs/2
    lo = max((f0 - bw/2)/ny, 1e-8); hi = min((f0 + bw/2)/ny, 0.999999)
    sos = DSP.Filters.digitalfilter(DSP.Filters.Bandstop(lo, hi), DSP.Filters.Butterworth(order))
    return DSP.filtfilt(sos, collect(Float64, x))
end

line_length(x) = sum(abs.(diff(x))) / max(1, length(x))
rmsval(x) = sqrt(mean(abs2, x))
variance(x) = var(x)

function band_power(x, fs, low, high)
    y = bandpass_filter(x, fs, low, high)
    return mean(abs2, y)
end

#sliding window seizure detector 
function detect_seizures(signal::AbstractVector{<:Real}, fs::Real;
                         win_sec::Real=2.0, step_sec::Real=0.5,
                         mains_freq::Real=50.0,
                         score_threshold::Real=3.0,
                         weights = (delta=1.0, line=1.0, var=0.6))
                         
    x = bandpass_filter(signal, fs, 0.5, min(90.0, fs/2 - 1.0))
    x = notch_filter(x, fs; f0=mains_freq)

    win = Int(round(win_sec * fs)); step = Int(round(step_sec * fs))
    n = length(x)
    if n < win
        return Vector{Tuple{Float64,Float64,Float64}}(), (scores = Float64[], times = Float64[])
    end
    starts = 1:step:(n - win + 1)
    nw = length(starts)

    deltas = zeros(nw); lines = zeros(nw); vars = zeros(nw)
    for (i, s) in enumerate(starts)
        w = @view x[s:s+win-1]
        deltas[i] = band_power(w, fs, 0.5, 4.0)
        lines[i] = line_length(w)
        vars[i] = variance(w)
    end

    zscore_f(v) = (v .- mean(v)) ./ (std(v) + eps())
    Zd = zscore_f(deltas); Zl = zscore_f(lines); Zv = zscore_f(vars)
    score = weights.delta*Zd .+ weights.line*Zl .+ weights.var*Zv

    is_pos = score .> score_threshold
    intervals = Vector{Tuple{Float64,Float64,Float64}}()
    i = 1
    while i <= nw
        if is_pos[i]
            j = i
            while j+1 <= nw && is_pos[j+1]
                j += 1
            end
            start_time = (starts[i]-1)/fs
            end_time   = (starts[j]-1 + win)/fs
            avg_score = mean(score[i:j])
            push!(intervals, (start_time, end_time, avg_score))
            i = j + 1
        else
            i += 1
        end
    end

    return intervals, (scores = score, times = (starts .- 1)./fs, filtered_signal = x)
end

#example usage that takes in file argument
function run_file()
    if length(ARGS) < 1
        println("Usage: julia seizure_detection.edfPlus.jl /path/to/file.edf")
        return
    end
    path = ARGS[1]
    chans = load_edf_channels(path)
    println("Loaded channels: ", keys(chans))
    label = first(keys(chans))
    signal, fs = chans[label]

    intervals, meta = detect_seizures(signal, fs)

    #print intervals
    for (s,e,sc) in intervals
        println("Possible seizure on '$label': $(round(s,digits=2))s - $(round(e,digits=2))s  score=$(round(sc,digits=2))")
    end

    #plots the signal with highlighted seizure intervals
    t = (0:length(meta.filtered_signal)-1) ./ fs
    plt = plot(t, meta.filtered_signal, label="EEG signal ($label)", color=:blue)
    for (s,e,_) in intervals
        plot!(plt, [s,e], [maximum(meta.filtered_signal)*1.1, maximum(meta.filtered_signal)*1.1], lw=6, color=:red, label=false)
    end
    xlabel!("Time (s)")
    ylabel!("Voltage")
    title!("EEG with Detected Seizures")
    display(plt)
    println("Press Enter to exit:")
    readline()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_file()
end
