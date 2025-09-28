"""GPU Utilities for RTX 4060 CUDA 12.4 Optimization - AMIL Project"""

import torch
import logging
import psutil
import time
from typing import Dict, Optional, Tuple
import subprocess
import json
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUMemoryInfo:
    """GPU memory information structure"""
    total: int
    used: int
    free: int
    utilization: float

@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics structure"""
    temperature: float
    power_usage: float
    memory_clock: float
    graphics_clock: float
    utilization: float

class GPUManager:
    """Advanced GPU management and optimization for RTX 4060 CUDA 12.4"""

    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.cuda_available = False
        self.monitoring_active = False
        self.performance_history = []
        self._lock = threading.Lock()

        self.initialize_gpu()

    def initialize_gpu(self):
        """Initialize GPU and detect capabilities"""
        logger.info("🔍 Detecting GPU capabilities...")

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.device = torch.device("cuda:0")
            self._detect_gpu_info()
            self._optimize_gpu_settings()
            self._verify_cuda_version()
            logger.info(f"✅ GPU initialized: {self.gpu_info.get('name', 'Unknown GPU')}")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️  CUDA not available, using CPU fallback")

    def _detect_gpu_info(self):
        """Detect detailed GPU information"""
        try:
            if not self.cuda_available:
                return

            # Basic PyTorch GPU info
            gpu_properties = torch.cuda.get_device_properties(0)

            self.gpu_info = {
                'name': gpu_properties.name,
                'compute_capability': f"{gpu_properties.major}.{gpu_properties.minor}",
                'total_memory_gb': round(gpu_properties.total_memory / (1024**3), 2),
                'multiprocessor_count': gpu_properties.multi_processor_count,
                'max_threads_per_multiprocessor': gpu_properties.max_threads_per_multi_processor,
                'max_threads_per_block': gpu_properties.max_threads_per_block,
                'warp_size': gpu_properties.warp_size,
                'cuda_cores': self._estimate_cuda_cores(gpu_properties),
                'memory_bandwidth': self._estimate_memory_bandwidth(gpu_properties),
                'tensor_cores': self._detect_tensor_cores(gpu_properties)
            }

            # Try to get additional info via nvidia-ml-py or nvidia-smi
            self._get_extended_gpu_info()

        except Exception as e:
            logger.error(f"Failed to detect GPU info: {str(e)}")

    def _optimize_gpu_settings(self):
        """Apply RTX 4060 specific optimizations for CUDA 12.4"""
        if not self.cuda_available:
            return

        try:
            # Enable cuDNN optimizations
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Enable TensorFloat-32 (TF32) for RTX 4060
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of 8GB

            # Enable memory pool for better allocation
            torch.cuda.empty_cache()

            # Set optimal device for multi-GPU systems (if applicable)
            torch.cuda.set_device(0)

            logger.info("🚀 RTX 4060 CUDA 12.4 optimizations applied")
            logger.info(f"   - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
            logger.info(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            logger.info(f"   - Memory fraction: 85%")

        except Exception as e:
            logger.error(f"Failed to apply GPU optimizations: {str(e)}")

    def _verify_cuda_version(self):
        """Verify CUDA 12.4 compatibility"""
        try:
            cuda_version = torch.version.cuda
            pytorch_version = torch.__version__

            logger.info(f"🔧 CUDA Version: {cuda_version}")
            logger.info(f"🔧 PyTorch Version: {pytorch_version}")

            # Check if we have the expected CUDA 12.4
            if cuda_version and cuda_version.startswith('12.'):
                logger.info("✅ CUDA 12.x detected - compatible with RTX 4060")
            else:
                logger.warning(f"⚠️  CUDA version {cuda_version} may not be optimal")

            self.gpu_info.update({
                'cuda_version': cuda_version,
                'pytorch_version': pytorch_version,
                'cuda_12_compatible': cuda_version.startswith('12.') if cuda_version else False
            })

        except Exception as e:
            logger.error(f"Failed to verify CUDA version: {str(e)}")

    def _estimate_cuda_cores(self, props) -> int:
        """Estimate CUDA cores based on GPU architecture"""
        # RTX 4060 has 3072 CUDA cores
        if "RTX 4060" in props.name:
            return 3072
        elif "RTX 40" in props.name:
            # Rough estimation for other RTX 40 series
            return props.multi_processor_count * 128
        else:
            # Generic estimation
            return props.multi_processor_count * 64

    def _estimate_memory_bandwidth(self, props) -> str:
        """Estimate memory bandwidth"""
        if "RTX 4060" in props.name:
            return "272 GB/s"  # RTX 4060 memory bandwidth
        else:
            return "Unknown"

    def _detect_tensor_cores(self, props) -> bool:
        """Detect if GPU has Tensor Cores"""
        # RTX 4060 has 3rd gen RT cores and 4th gen Tensor cores
        return "RTX" in props.name and props.major >= 7

    def _get_extended_gpu_info(self):
        """Get extended GPU information using nvidia-smi"""
        try:
            # Try nvidia-ml-py first
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Get additional info
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                self.gpu_info.update({
                    'temperature_c': temperature,
                    'power_usage_w': power_usage,
                    'memory_used_gb': round(memory_info.used / (1024**3), 2),
                    'memory_free_gb': round(memory_info.free / (1024**3), 2),
                    'gpu_utilization': utilization.gpu,
                    'memory_utilization': utilization.memory
                })

            except ImportError:
                # Fallback to nvidia-smi
                self._get_nvidia_smi_info()

        except Exception as e:
            logger.warning(f"Could not get extended GPU info: {str(e)}")

    def _get_nvidia_smi_info(self):
        """Get GPU info using nvidia-smi command"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) == 5:
                    self.gpu_info.update({
                        'temperature_c': float(values[0]),
                        'power_usage_w': float(values[1]),
                        'memory_used_mb': float(values[2]),
                        'memory_free_mb': float(values[3]),
                        'gpu_utilization': float(values[4])
                    })

        except Exception as e:
            logger.warning(f"nvidia-smi not available: {str(e)}")

    def get_memory_info(self) -> GPUMemoryInfo:
        """Get current GPU memory information"""
        if not self.cuda_available:
            return GPUMemoryInfo(0, 0, 0, 0.0)

        try:
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_cached = torch.cuda.memory_cached(0) if hasattr(torch.cuda, 'memory_cached') else 0

            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = total_memory - memory_reserved
            utilization = (memory_allocated / total_memory) * 100

            return GPUMemoryInfo(
                total=total_memory,
                used=memory_allocated,
                free=free_memory,
                utilization=utilization
            )

        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {str(e)}")
            return GPUMemoryInfo(0, 0, 0, 0.0)

    def get_performance_metrics(self) -> GPUPerformanceMetrics:
        """Get current GPU performance metrics"""
        if not self.cuda_available:
            return GPUPerformanceMetrics(0, 0, 0, 0, 0)

        try:
            # Update extended info
            self._get_extended_gpu_info()

            return GPUPerformanceMetrics(
                temperature=self.gpu_info.get('temperature_c', 0),
                power_usage=self.gpu_info.get('power_usage_w', 0),
                memory_clock=0,  # Would need additional tools
                graphics_clock=0,  # Would need additional tools
                utilization=self.gpu_info.get('gpu_utilization', 0)
            )

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return GPUPerformanceMetrics(0, 0, 0, 0, 0)

    def optimize_for_inference(self):
        """Optimize GPU settings for inference workload"""
        if not self.cuda_available:
            return

        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Set to eval mode optimizations
            torch.backends.cudnn.benchmark = True

            # Disable gradient computation globally if not already done
            torch.set_grad_enabled(False)

            logger.info("🎯 GPU optimized for inference workload")

        except Exception as e:
            logger.error(f"Failed to optimize for inference: {str(e)}")

    def optimize_for_training(self):
        """Optimize GPU settings for training workload"""
        if not self.cuda_available:
            return

        try:
            # Enable gradient computation
            torch.set_grad_enabled(True)

            # Mixed precision training setup
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Clear cache for training
            torch.cuda.empty_cache()

            logger.info("🎯 GPU optimized for training workload")

        except Exception as e:
            logger.error(f"Failed to optimize for training: {str(e)}")

    def profile_operation(self, operation_name: str):
        """Context manager for profiling GPU operations"""
        return GPUProfiler(self, operation_name)

    def benchmark_gpu(self) -> Dict:
        """Benchmark GPU performance with representative operations"""
        if not self.cuda_available:
            return {'error': 'CUDA not available'}

        logger.info("🏃‍♂️ Running GPU benchmark...")

        benchmark_results = {}

        try:
            # Matrix multiplication benchmark
            with self.profile_operation("matrix_multiplication"):
                size = 2048
                a = torch.randn(size, size, device=self.device, dtype=torch.float32)
                b = torch.randn(size, size, device=self.device, dtype=torch.float32)

                start_time = time.time()
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()

                matrix_mult_time = (time.time() - start_time) / 10
                benchmark_results['matrix_multiplication_ms'] = round(matrix_mult_time * 1000, 2)

            # Memory bandwidth test
            with self.profile_operation("memory_bandwidth"):
                size = 100_000_000  # 100M elements
                data = torch.randn(size, device=self.device, dtype=torch.float32)

                start_time = time.time()
                for _ in range(5):
                    _ = data + 1.0
                torch.cuda.synchronize()

                memory_time = (time.time() - start_time) / 5
                benchmark_results['memory_bandwidth_ms'] = round(memory_time * 1000, 2)

            # Tensor operations benchmark
            with self.profile_operation("tensor_operations"):
                data = torch.randn(1000, 1000, device=self.device)

                start_time = time.time()
                for _ in range(100):
                    result = torch.relu(torch.matmul(data, data.T))
                torch.cuda.synchronize()

                tensor_ops_time = (time.time() - start_time) / 100
                benchmark_results['tensor_operations_ms'] = round(tensor_ops_time * 1000, 2)

            # Memory info during benchmark
            memory_info = self.get_memory_info()
            benchmark_results['memory_utilization'] = round(memory_info.utilization, 2)
            benchmark_results['memory_used_gb'] = round(memory_info.used / (1024**3), 2)

            logger.info("✅ GPU benchmark completed")

        except Exception as e:
            logger.error(f"GPU benchmark failed: {str(e)}")
            benchmark_results['error'] = str(e)

        return benchmark_results

    def get_device_info(self) -> Dict:
        """Get comprehensive device information"""
        device_info = {
            'cuda_available': self.cuda_available,
            'device': str(self.device),
            'gpu_count': torch.cuda.device_count() if self.cuda_available else 0,
        }

        if self.cuda_available:
            device_info.update(self.gpu_info)

            # Add current memory info
            memory_info = self.get_memory_info()
            device_info.update({
                'current_memory_used_gb': round(memory_info.used / (1024**3), 2),
                'current_memory_free_gb': round(memory_info.free / (1024**3), 2),
                'current_memory_utilization': round(memory_info.utilization, 2)
            })

        return device_info

    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use"""
        return self.cuda_available

    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage information"""
        if not self.cuda_available:
            return {'gpu_memory': 'N/A', 'system_memory': f"{psutil.virtual_memory().percent}%"}

        memory_info = self.get_memory_info()

        return {
            'gpu_memory_used_gb': round(memory_info.used / (1024**3), 2),
            'gpu_memory_total_gb': round(memory_info.total / (1024**3), 2),
            'gpu_memory_utilization': f"{memory_info.utilization:.1f}%",
            'system_memory_utilization': f"{psutil.virtual_memory().percent}%",
            'system_memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }

    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            logger.debug("Peak memory stats reset")

    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous GPU monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor():
            while self.monitoring_active:
                try:
                    metrics = self.get_performance_metrics()
                    memory_info = self.get_memory_info()

                    with self._lock:
                        self.performance_history.append({
                            'timestamp': time.time(),
                            'temperature': metrics.temperature,
                            'power_usage': metrics.power_usage,
                            'utilization': metrics.utilization,
                            'memory_utilization': memory_info.utilization
                        })

                        # Keep only last 100 entries
                        if len(self.performance_history) > 100:
                            self.performance_history.pop(0)

                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"GPU monitoring error: {str(e)}")
                    time.sleep(interval_seconds)

        monitoring_thread = threading.Thread(target=monitor, daemon=True)
        monitoring_thread.start()

        logger.info(f"🔍 GPU monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring_active = False
        logger.info("🔍 GPU monitoring stopped")

    def get_monitoring_data(self) -> List[Dict]:
        """Get historical monitoring data"""
        with self._lock:
            return self.performance_history.copy()

class GPUProfiler:
    """Context manager for profiling GPU operations"""

    def __init__(self, gpu_manager: GPUManager, operation_name: str):
        self.gpu_manager = gpu_manager
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        if self.gpu_manager.cuda_available:
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_manager.cuda_available:
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_used = end_memory - self.start_memory
        else:
            memory_used = 0

        end_time = time.time()
        execution_time = end_time - self.start_time

        logger.debug(f"GPU Profile [{self.operation_name}]: "
                     f"{execution_time*1000:.2f}ms, "
                     f"Memory: {memory_used/(1024**2):.2f}MB")

# Utility functions
def get_optimal_batch_size(model_size_mb: float, available_memory_gb: float) -> int:
    """Calculate optimal batch size based on model size and available memory"""
    # Rule of thumb: use 70% of available memory
    usable_memory_mb = available_memory_gb * 1024 * 0.7

    # Estimate memory per sample (rough approximation)
    memory_per_sample = model_size_mb * 0.1  # 10% of model size per sample

    if memory_per_sample > 0:
        batch_size = int(usable_memory_mb / memory_per_sample)
        return max(1, min(batch_size, 64))  # Clamp between 1 and 64

    return 16  # Default fallback

def check_gpu_compatibility() -> Dict:
    """Check GPU compatibility for the project"""
    compatibility_report = {
        'cuda_available': torch.cuda.is_available(),
        'recommended_gpu': 'RTX 4060 or better',
        'minimum_memory_gb': 8,
        'recommended_cuda_version': '12.4',
        'compatible': False,
        'warnings': [],
        'recommendations': []
    }

    if not torch.cuda.is_available():
        compatibility_report['warnings'].append("CUDA not available - will use CPU fallback")
        compatibility_report['recommendations'].append("Install CUDA 12.4 and compatible PyTorch")
        return compatibility_report

    # Check GPU memory
    gpu_props = torch.cuda.get_device_properties(0)
    memory_gb = gpu_props.total_memory / (1024**3)

    if memory_gb < 8:
        compatibility_report['warnings'].append(f"GPU memory ({memory_gb:.1f}GB) below recommended 8GB")

    # Check CUDA version
    cuda_version = torch.version.cuda
    if cuda_version and not cuda_version.startswith('12.'):
        compatibility_report['warnings'].append(f"CUDA version {cuda_version} may not be optimal")
        compatibility_report['recommendations'].append("Consider upgrading to CUDA 12.4")

    # Check compute capability
    if gpu_props.major < 7:
        compatibility_report['warnings'].append("GPU compute capability below 7.0 - may lack modern features")

    # Overall compatibility
    compatibility_report['compatible'] = (
            memory_gb >= 6 and  # Minimum viable memory
            gpu_props.major >= 6  # Minimum compute capability
    )

    if compatibility_report['compatible'] and not compatibility_report['warnings']:
        compatibility_report['recommendations'].append("GPU setup is optimal for this project")

    return compatibility_report
