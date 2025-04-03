function prepare_sampler(
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm},
    chains::Chains,
)
    @error "Save and continue for sampler type $(typeof(sampler)) not implemented"
end

# prepare sampler for nuts
function prepare_sampler(sampler::NUTS, chains::Chains)
    # get the last step size
    sampler_ϵ = chains[:step_size][end]
    # create a new sampler with the last state
    # n_adapts can be zero because we load the warmed-up sampler state
    return NUTS(
        0,
        sampler.δ;
        Δ_max = sampler.Δ_max,
        adtype = sampler.adtype,
        init_ϵ = sampler_ϵ,
        max_depth = sampler.max_depth,
    )
end

function validate_saved_sampling_state!(
    save_resume::ChainSaveResume,
    n_segments::Int,
    n_chains::Int,
)
    # check if the path exists
    if !isdir(save_resume.path)
        @warn "Path $(save_resume.path) does not exist, creating it"
        mkdir(save_resume.path)
    end
    # check if the path is a directory
    if !isdir(save_resume.path)
        @error "Path $(save_resume.path) is not a directory"
    end
    # check if the path is writable
    if !(uperm(save_resume.path) & 0x02 == 0x02)
        @error "Path $(save_resume.path) is not writable"
    end


    # find the last segment (for each chain)
    last_segment = Int[]
    for chain = 1:n_chains
        last_seg = 0
        n_segs = 0
        for cur_seg = 1:n_segments
            if isfile(
                joinpath(
                    save_resume.path,
                    "$(save_resume.chain_prefix)_c$(chain)_s$(cur_seg).h5",
                ),
            )
                last_seg = cur_seg
                n_segs += 1
            end
        end
        if n_segs < last_seg
            @error "Chain $chain has missing segments, check the path $(save_resume.path)"
        end
        push!(last_segment, last_seg)
    end

    return last_segment
end

function load_segment(save_resume::ChainSaveResume, chain_n::Int, segment::Int)
    # load the chain
    chain = h5open(
        joinpath(
            save_resume.path,
            "$(save_resume.chain_prefix)_c$(chain_n)_s$(segment).h5",
        ),
        "r",
    ) do file
        read(file, Chains)
    end
    # extra validation?
    return chain
end

function save_segment(seg::Chains, save_resume::ChainSaveResume, chain_n::Int, seg_n::Int)
    # save the chain
    h5open(
        joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain_n)_s$(seg_n).h5"),
        "w",
    ) do file
        write(file, seg)
    end
end

function combine_segments(save_resume::ChainSaveResume, n_segments::Int, n_chains::Int)
    chains::Vector{Union{Nothing,Chains}} = fill(nothing, n_chains)
    for chain = 1:n_chains
        segments::Vector{Union{Nothing,Chains}} = fill(nothing, n_segments)
        seg_start = 1
        for segment = 1:n_segments
            seg = load_segment(save_resume, chain, segment)
            # update the range
            seg_end = seg_start + length(seg) - 1
            seg = setrange(seg, seg_start:seg_end)
            seg_start = seg_end + 1
            segments[segment] = seg
        end
        chains[chain] = cat(segments..., dims = 1)
    end

    return chainscat(chains...)
end

