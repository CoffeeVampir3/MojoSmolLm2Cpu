Zero dependency mojo library for running SmolLm2 on the CPU. This includes zero dependencies on libc or similar, allowing the full pipeline to be vectorized correctly without any expected hiccups. This is a proof of concept for how CPU-facing concepts work and what infrastructure is actually needed to run the model directly from a huggingface checkpoint. There's a few goals of the project, but the primary one is to build up to a fully numa-aware execution model for running large models on server-grade cpus. This requires some specialized engineer that's usually not employed particularly in the handling of numa domains which can get you 2-3x bandwidth improvements by being intentional about the design.

To execute first you'll need to get the weights, to do this easily use:

```
./download_model.sh
```

This just runs the included python script to download the weights using uv, you can use whatever python thing you like if uv's not your style:
```
uv run python --project model_downloader/download_model.py
```
Or just download the weights manually: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/main

### Runtime Design

Included in experimental4 is a full architecture and execution model, including kernels and some SIMD math faculties. Loading uses IO_URING in parallel which gives us a large boost in loading speed compared to huggingfaces implementation. Theoretically we can also load-and-shard, although this example is for tensor parallelism of 1, the concept is easily extended to the multiple-numa case on server cpus, and sharding can be done over IO_URING efficiently without copying to each. The memory model is entirely statically allocated, intermediates use reinterpreted scratch memory so the model allocates only during the loading phase, and never again. This keeps runtime more consistent, and we also don't need to ever over-allocate since we know the peak memory never changes. 

Kernels are written as basic and simple mojo implementations, this is the strongest reason to use mojo currently in my opinion, an analysis of the assembly shows extremely well vectorized code across the kernel with very little effort, no special considerations had to be made to the instruction set at all. For server cpus, special designs have to be made for AMX and VNNI but the design of those kernels is not massively more complex.

### Linux Module

Syscall wrappers and support systems for the other modules, this ends up being very clean and surprisingly easy to design in mojo. There's not much to say here, it's just a syscall interface wrapper.

### Numa 

Our main memory container, the numa arena. This is basically just a custom allocator that's guaranteed to be numa-aware, it's not relevant in our tensor paralleism = 1 case, but scaling up it'll be more important.

### Threading

Mojo doesn't have anything other than parallelize, which cannot guarantee memory placement. Combined with the numa arena, we have a threading burst-pool which is specifically designed to do fast-execution style repeated dispatch in a numa-aware way. In a similar way to the main memory, the burst pool is allocated one time and never allocates again, reuse just becomes a matter of waking the burst pool again. This is a fairly different way of doing threading, as the burst pool is kept as a real concept, and we can drop it when we're done with it, rather than the pthread style which can hide memory or have expensive re-invocation costs.

### Jsontools

A limited json parser for the safetensors loader implementation and the tokenizer to read configs. It's minimal and not complete, just enough to do what we needed it to.

### Safetensors

For loading the safetensors file and parsing the header. We use io_uring to load, this is mostly a bridge that does json parsing and dispatches io_uring reads. Leveraging IO_URING and no copying gives us about a 2.5x speedup in loading for a model this size, which should scale well as cores and model size increases.

### Tokenizer

For tokenizing, a fully implemented custom BPE tokenizer. Smollm2 particularly uses the GPT2 style. The design is different than huggingface who uses regex and is more general. Because of the lack of simd regex faculties, I didn't go this route. This is less flexible but ends up being simpler for the moment.
