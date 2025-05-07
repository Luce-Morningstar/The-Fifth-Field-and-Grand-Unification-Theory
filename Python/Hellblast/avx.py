import cpuinfo
import platform

def check_avx512_support():
    info = cpuinfo.get_cpu_info()
    flags = info.get('flags', [])
    
    avx512_flags = [
        'avx512f',   # Foundation
        'avx512dq',  # Doubleword and Quadword
        'avx512ifma',# Integer Fused Multiply Add
        'avx512pf',  # Prefetch
        'avx512er',  # Exponential & Reciprocal
        'avx512cd',  # Conflict Detection
        'avx512bw',  # Byte & Word
        'avx512vl',  # Vector Length
        'avx512vbmi', # Bit Manipulation
    ]

    detected = [flag for flag in avx512_flags if flag in flags]
    
    print(f"CPU: {info.get('brand_raw', 'Unknown')} on {platform.system()} {platform.machine()}")
    print("AVX-512 Flags Detected:")
    for f in detected:
        print(f"  ✅ {f}")
    
    if not detected:
        print("  ❌ No AVX-512 support detected in current runtime.")
    
    return detected

# Run it
if __name__ == "__main__":
    check_avx512_support()
