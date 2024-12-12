import torch
import time
from naive_attention import manual_attention

def benchmark_cuda_function(func, *args, num_runs=10, warmup_runs=5):
    for _ in range(warmup_runs):
        func(*args)
        torch.cuda.synchronize() 

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        start_event.record() 
        func(*args)           
        end_event.record()    
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings.append(elapsed_time)
        
    avg_time = sum(timings) / len(timings)
    return avg_time


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description="Benchmark CUDA operations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the tensors")
    parser.add_argument("--tensor_size", type=int, default=1024, help="Size of the square tensors")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of warm-up runs")

    args = parser.parse_args()

    # Parameters from command-line arguments
    batch_size = args.batch_size
    tensor_size = args.tensor_size
    num_runs = args.num_runs
    warmup_runs = args.warmup_runs


    # Random CUDA tensors of shape (batch_size, tensor_size, tensor_size)
    a = torch.randn(batch_size, tensor_size, tensor_size, device='cuda')
    b = torch.randn(batch_size, tensor_size, tensor_size, device='cuda')

    # Benchmark the batched CUDA matrix multiplication
    avg_time_ms = benchmark_cuda_function(cuda_batched_matmul, a, b, num_runs=num_runs, warmup_runs=warmup_runs)
    print(f"Average execution time for batch size {batch_size} and tensor size {tensor_size}: {avg_time_ms:.4f} ms")
