#single-file robust EEG read / plot / filter script
using EDF, DataFrames, Plots, DSP, Statistics

# ------------------------
#helper functions
# ------------------------
function load_edf(path::AbstractString)
    f = EDF.read(path)
    EDF.read!(f)
    return f
end

function get_data_signals(f::EDF.File)
    return filter(s -> !(s isa EDF.AnnotationsSignal), f.signals)
end

function decode_all_channels(data_signals)
    # decode and return Vector{Vector{Float64}} of channels
    channels = Vector{Vector{Float64}}(undef, length(data_signals))
    for (i, s) in enumerate(data_signals)
        channels[i] = collect(EDF.decode(s))
    end
    return channels
end

function compute_fs(sig, header)
    return sig.header.samples_per_record / header.seconds_per_record
end

#safe integer sample index for seconds
seconds_to_samples(seconds::Real, fs::Real) = floor(Int, seconds * fs)

#simple design: notch (bandstop) around f0 with quality factor Q
function design_notch(fs::Real, f0::Real=60.0, q::Real=30.0, order::Int=2)
    nyq = fs/2
    bw = f0 / q              # Δf = f0/Q
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

#parse EDF+ annotations signal into tuples of (onset_s, duration_s, text)
function parse_annotations(f::EDF.File)
    idx = findfirst(s -> s isa EDF.AnnotationsSignal, f.signals)
    if idx === nothing
        return []
    end
    s = f.signals[idx]
    anns = []
    if hasproperty(s, :records)
        for rec in s.records
            # Many implementations: rec.annotations is a vector of (onset, duration, text)
            if hasproperty(rec, :annotations)
                for a in rec.annotations
                    onset = getproperty(a, :onset, nothing)
                    dur   = getproperty(a, :duration, 0.0)
                    txt   = getproperty(a, :text, "")
                    push!(anns, (onset, dur, txt))
                end
            else
                # fallback: try fields directly on rec
                try
                    push!(anns, (rec.onset, rec.duration, rec.text))
                catch
                    push!(anns, ("unknown", 0.0, sprint(show, rec)))
                end
            end
        end
    end
    return anns
end

#overlay shaded rectangles (annotations) on a plot
function overlay_annotations!(plt, anns, ymin, ymax; alpha=0.15, color=:red)
    for (onset, dur, txt) in anns
        if onset === nothing || onset == "unknown" continue end
        x_rect = [onset, onset, onset+dur, onset+dur]
        y_rect = [ymin, ymax, ymax, ymin]
        plot!(plt, x_rect, y_rect, seriestype=:shape, fillalpha=alpha, linealpha=0, fillcolor=color)
    end
end

# Keep plot displayed until Enter is pressed
function wait_for_enter(message="Press Enter to continue...")
    println(message)
    readline()
end

# ------------------------
#main usage
# ------------------------
#adjust path and parameters here
edf_path = "/Users/joshkweon/Desktop/csci_6221_julia/data/seizure/SZ_4.edf"
f = load_edf(edf_path)

#list signals
println("All signals:")
for (i, s) in enumerate(f.signals)
    if s isa EDF.AnnotationsSignal
        println("[$i] <AnnotationsSignal>")
    else
        println("[$i] ", s.header.label, "  samples_per_record=", s.header.samples_per_record)
    end
end

#get data channels, decode once
data_signals = get_data_signals(f)
channels = decode_all_channels(data_signals)

#compute fs from first channel and check others
fs = compute_fs(data_signals[1], f.header)
println("sampling rate inferred: ", fs, " Hz")
for (i, s) in enumerate(data_signals)
    fs_i = s.header.samples_per_record / f.header.seconds_per_record
    if abs(fs_i - fs) > 1e-6
        @warn "Channel $i has different declared sampling rate: $fs_i (using first channel fs=$fs)"
    end
end

#simple single-channel plot example (first channel)
seconds_to_plot = 10.0
n = min(length(channels[1]), seconds_to_samples(seconds_to_plot, fs))
t = collect(0.0:(1.0/fs):((n-1)/fs))

#downsample for display if needed
max_points = 2000
step = max(1, floor(Int, length(t)/max_points))

p = plot(t[1:step:end], channels[1][1:step:n],
         xlabel="Time (s)", ylabel="µV", title="Channel: $(data_signals[1].header.label)",
         legend=false)

#overlay annotations (if present)
anns = parse_annotations(f)
if !isempty(anns)
    ymin, ymax = extrema(channels[1][1:n])
    overlay_annotations!(p, anns, ymin - 50, ymax + 50)  # add margin
end

display(p)
wait_for_enter("Press Enter to continue to filtered view:")

# ------------------------
#filtering example: notch -> bandpass applied to all channels
# ------------------------
notch_filter = design_notch(fs, 60.0, 30.0, 2)
bp_filter = design_bandpass(fs, 0.5, 70.0, 4)

filtered_channels = Vector{Vector{Float64}}(undef, length(channels))
for i in 1:length(channels)
    filtered_channels[i] = apply_filters(channels[i], fs;
                                         notch_filter=notch_filter, bp_filter=bp_filter)
end

#show pre vs post for first channel
n2 = min(length(channels[1]), seconds_to_samples(5, fs))
t2 = collect(0.0:(1.0/fs):((n2-1)/fs))
p2 = plot(t2[1:step:end], channels[1][1:step:n2], label="raw")
plot!(p2, t2[1:step:end], filtered_channels[1][1:step:n2], label="filtered")
title!(p2, "Raw vs Filtered (first channel)")
display(p2)
wait_for_enter("Press Enter to exit:")