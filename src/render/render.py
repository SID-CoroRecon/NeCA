import torch
import torch.nn as nn
import gc

def run_network(inputs, fn, netchunk):
    """
    Prepares inputs and applies network "fn" with memory optimization.
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    output_chunks = []
    
    # Process in chunks with memory management
    for i in range(0, uvt_flat.shape[0], netchunk):
        chunk = uvt_flat[i:i + netchunk]
        chunk_output = fn(chunk)
        output_chunks.append(chunk_output)
        
        # Periodic memory cleanup for large datasets
        if i > 0 and i % (netchunk * 4) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    out_flat = torch.cat(output_chunks, 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    
    # Clean up intermediate tensors
    del output_chunks, uvt_flat, out_flat
    
    return out

def run_network_memory_efficient(inputs, fn, netchunk, use_amp=True):
    """
    Memory-efficient version with automatic mixed precision and smaller chunks.
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    output_chunks = []
    
    # Use smaller chunk size for better memory efficiency
    effective_chunk_size = min(netchunk, uvt_flat.shape[0] // 8) if uvt_flat.shape[0] > netchunk else netchunk
    
    for i in range(0, uvt_flat.shape[0], effective_chunk_size):
        chunk = uvt_flat[i:i + effective_chunk_size]
        
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                chunk_output = fn(chunk)
        else:
            chunk_output = fn(chunk)
            
        output_chunks.append(chunk_output.detach() if chunk_output.requires_grad else chunk_output)
        
        # More frequent memory cleanup for very large inputs
        if i % effective_chunk_size == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    out_flat = torch.cat(output_chunks, 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    
    # Cleanup
    del output_chunks, uvt_flat, out_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return out 