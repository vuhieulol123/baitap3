import cv2
import numpy as np
import matplotlib.pyplot as plt


def _distance_matrix(shape):
    """Ma trận khoảng cách D(u,v) tâm tại giữa ảnh."""
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U**2 + V**2)

def ideal_hpf(shape, D0):
    D = _distance_matrix(shape)
    return (D > D0).astype(float)

def gaussian_hpf(shape, D0):
    D = _distance_matrix(shape)
    return 1.0 - np.exp(-(D**2) / (2.0 * (D0**2) + 1e-8))

def butterworth_hpf(shape, D0, n=2):
    D = _distance_matrix(shape)
    eps = 1e-8
    return 1.0 / (1.0 + (D0 / (D + eps))**(2 * n))

def apply_hpf_fft(img_gray, H):
    """Áp dụng H(u,v) trong miền tần số và trả về ảnh biên (magnitude)."""
    F = np.fft.fft2(img_gray)
    F_shift = np.fft.fftshift(F)
    G_shift = F_shift * H
    G = np.fft.ifftshift(G_shift)
    g = np.fft.ifft2(G)
    return np.abs(g)

def normalize_uint8(x):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)

# ======================= Hậu xử lý, đếm đối tượng =======================

def otsu_binarize(edge_img):
    edge8 = normalize_uint8(edge_img)
    _, th = cv2.threshold(edge8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def postprocess_edges(binary, morph_iter=1):
    """Đóng + dày biên một chút để nối nét đứt (giúp đếm chính xác hơn)."""
    if morph_iter <= 0:
        return binary
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=morph_iter)
    binary = cv2.dilate(binary, k, iterations=1)
    return binary

def count_components(binary, min_area=40):
    """Đếm thành phần liên thông, bỏ nhiễu theo diện tích tối thiểu."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    return len(keep), labels, stats, centroids, mask

#Pipeline chính 

def run_pipeline(img_gray, cutoff_ratio=0.10, butter_n=2, morph_iter=1, min_area=40):
    """
    cutoff_ratio: D0 = cutoff_ratio * min(H, W)
    butter_n: bậc Butterworth
    morph_iter: số lần đóng/dilate để nối biên
    min_area: ngưỡng bỏ nhiễu khi đếm
    """
    Hh, Ww = img_gray.shape
    D0 = cutoff_ratio * min(Hh, Ww)

    filters = {
        "Ideal HPF": ideal_hpf(img_gray.shape, D0),
        "Gaussian HPF": gaussian_hpf(img_gray.shape, D0),
        f"Butterworth HPF (n={butter_n})": butterworth_hpf(img_gray.shape, D0, n=butter_n),
    }

    results = {}
    for name, H in filters.items():
        edge = apply_hpf_fft(img_gray, H)
        binary = otsu_binarize(edge)
        binary = postprocess_edges(binary, morph_iter=morph_iter)
        count, labels, stats, cents, mask = count_components(binary, min_area=min_area)
        results[name] = {
            "edge": normalize_uint8(edge),
            "binary": binary,
            "count": count,
            "labels": labels,
            "mask": mask,
        }
    return results


if __name__ == "__main__":

    # 1) 
    img = cv2.imread("anhdt.jpg", cv2.IMREAD_GRAYSCALE)
    gray = img
    # 2) 
    results = run_pipeline(
        gray,
        cutoff_ratio=0.05,  
        butter_n=2,
        morph_iter=1,
        min_area=40
    )
    # 3) 
    print("Số đối tượng (connected components) theo từng bộ lọc:")
    for k, v in results.items():
        print(f"  - {k}: {v['count']}")

    # 4) 
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 3, 1); plt.title("Ảnh gốc (grayscale)"); plt.imshow(gray, cmap='gray'); plt.axis('off')

    names = list(results.keys())
    plt.subplot(2, 3, 2); plt.title(f"{names[0]} - biên");   plt.imshow(results[names[0]]['edge']);   plt.axis('off')
    plt.subplot(2, 3, 3); plt.title(f"{names[1]} - biên");   plt.imshow(results[names[1]]['edge']);   plt.axis('off')
    plt.subplot(2, 3, 4); plt.title(f"{names[2]} - biên");   plt.imshow(results[names[2]]['edge']);   plt.axis('off')
    plt.subplot(2, 3, 5); plt.title(f"{names[0]} - nhị phân"); plt.imshow(results[names[0]]['binary']); plt.axis('off')
    plt.subplot(2, 3, 6); plt.title(f"{names[1]} - nhị phân"); plt.imshow(results[names[1]]['binary']); plt.axis('off')
    plt.tight_layout()
    plt.show()
