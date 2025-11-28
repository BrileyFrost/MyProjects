import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from skimage import data, color, filters, util
from skimage.transform import resize, rescale
from skimage.restoration import denoise_bilateral
import cv2


# --- Extra Functions --- #
def steerable_basis(img, sigma=1.0):
    Ix = gaussian_filter1d(img, sigma=sigma, order=1, axis=1)
    Iy = gaussian_filter1d(img, sigma=sigma, order=1, axis=0)
    return Ix, Iy
def steer_orientation(Ix, Iy, theta):
    return np.cos(theta) * Ix + np.sin(theta) * Iy
def steerable_pyramid(img, sigmas=[1.0, 2.0, 4.0], orientations=3,norm_sigma=3.0, alpha=1.0, beta=0.5, eps=1e-3):
    img = img.astype(float) / 255.0
    pyr = []
    for sigma in sigmas:
        Ix, Iy = steerable_basis(img, sigma=sigma)
        scale_responses = []
        for k in range(orientations):
            theta = np.pi * k / orientations
            R = steer_orientation(Ix, Iy, theta)
            scale_responses.append(R)
        # at this point you can simply do pyr.append(scale_responses) and remove lines 27-35 all that is. Is a energy normalization function minimal changes just makes it a bit more human like
        energy = np.zeros_like(img)
        for R in scale_responses:
            energy += R**2
        G = gaussian_filter(energy, sigma=norm_sigma)
        scale_norm = []
        denom = eps + alpha + beta * G
        # Don't replace with vectorized instead of loop it is slower 
        for R in scale_responses:
            scale_norm.append(R / denom)
        pyr.append(scale_norm)
    return pyr
def multi_scale_coherence(gray, sigmas=[1.0, 2.0, 4.0]):
    gx = filters.sobel_h(gray)
    gy = filters.sobel_v(gray)
    coh_scales = []
    for sigma in sigmas:
        Jxx = gaussian_filter(gx * gx, sigma)
        Jyy = gaussian_filter(gy * gy, sigma)
        Jxy = gaussian_filter(gx * gy, sigma)
        tmp = np.sqrt((Jxx - Jyy)**2 + 4 * (Jxy**2))
        lambda1 = (Jxx + Jyy + tmp) / 2.0
        lambda2 = (Jxx + Jyy - tmp) / 2.0
        coh = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
        coh = np.clip(coh, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        coh *= mag
        coh_scales.append(coh)
    return coh_scales
def weighted_coherence(pyr, coh_scales):
    weighted_maps = []
    for coh in coh_scales:
        responses = pyr
        mag = np.sqrt(np.sum([r**2 for r in responses], axis=0))
        weighted = coh * mag
        weighted_maps.append(weighted)
    return weighted_maps


# V1 Pipeline # 
def V1_Feature_extraction(gray,rgb,start=False,fovea_radius=100,peripheral_scale=0.25, sigmas=[1,2,4], Orientations=8, scale=0, ksize=18):
    # Still deciding on a proper disparity map implementation btw    
    h, w = gray.shape
    if not start:
        cy, cx = h//2, w//2
    else:
        cy, cx = start

    # Create fovea mask
    Y, X = np.ogrid[:h, :w]
    fovea_mask = (X-cx)**2 + (Y-cy)**2 <= fovea_radius**2
    periphery_mask = ~fovea_mask

    features = {}

    # -------- Fovea (full-res) --------
    pyr = steerable_pyramid(gray, sigmas=sigmas, orientations=Orientations)
    scales = np.array(pyr[scale])
    coh_scales = multi_scale_coherence(gray,sigmas)
    cohs = np.array(weighted_coherence(scales,coh_scales))
    texture = np.zeros_like(gray)
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequency = 0.25
    energy_maps = []
    for theta in orientations:
        g = cv2.getGaborKernel(
            (ksize, ksize),
            2.0,
            theta,
            1.0/frequency,
            0.5,
            0,
            ktype=cv2.CV_32F
        )
        r = cv2.filter2D(gray, cv2.CV_32F, g)
        energy_maps.append(r*r)
    stack = np.stack(energy_maps, axis=0)
    local_mean = cv2.blur(stack, (9,9))
    normalized = stack / (local_mean + 1e-6)
    normalized = normalized.sum(axis=0)
    texture[fovea_mask] = normalized[fovea_mask]

    # -------- Periphery (downsampled) --------
    gray_small = util.img_as_float(rescale(gray, peripheral_scale, anti_aliasing=True))
    pyr_p = steerable_pyramid(gray_small, sigmas=sigmas, orientations=Orientations)
    scales_p = np.array(pyr_p[scale])
    coh_scales_p = multi_scale_coherence(gray_small,sigmas)
    cohs_p = np.array(weighted_coherence(scales_p,coh_scales_p))
    energy_maps = []
    for theta in orientations:
        g = cv2.getGaborKernel(
            (ksize, ksize),
            2.0,
            theta,
            1.0/frequency,
            0.5,
            0,
            ktype=cv2.CV_32F
        )
        r = cv2.filter2D(gray_small, cv2.CV_32F, g)
        energy_maps.append(r*r)
    stack = np.stack(energy_maps, axis=0)
    local_mean = cv2.blur(stack, (9,9))
    normalized = stack / (local_mean + 1e-6)
    normalized = normalized.sum(axis=0)
    #local_mean_p = ndi.uniform_filter(gray_small, size=3)
    #local_var_p = ndi.uniform_filter((gray_small-local_mean_p)**2, size=3)

    # -------- Periphery Masks cause like numpy can't handle matrix operation unless they perfect dimensional shape ------- #
    # Technically speaking the periphery mask can be removed you would need to redesign function a bit though the only reason I do it is because I think it makes it look more human and I hope it will preserve time in later computations but if it doesn't I will remove it
    periphery_mask_b = np.broadcast_to(periphery_mask, cohs.shape)
    periphery_mask_c = np.broadcast_to(periphery_mask, scales.shape)

    # Upsample back to full size
    texture[periphery_mask] = resize(normalized, gray.shape, order=1, anti_aliasing=True)[periphery_mask]
    local_coh_p = resize(cohs_p,output_shape=cohs.shape,order=1,anti_aliasing=True,preserve_range=True)
    cohs[periphery_mask_b]=local_coh_p[periphery_mask_b]
    local_responses_p = resize(scales_p,output_shape=scales.shape,order=1,anti_aliasing=True,preserve_range=True)
    scales[periphery_mask_c] = local_responses_p[periphery_mask_c]


    # -------- Colors --------
    R = rgb[...,0]; G = rgb[...,1]; B = rgb[...,2]
    R_minus_G = (R-G+1)/2
    B_minus_Y = (B-(R+G)/2+1)/2


    # -------- Return the shit --------
    features['texture'] = texture
    features['R_minus_G'] = R_minus_G
    features['B_minus_Y'] = B_minus_Y
    features['scales'] = scales
    features['coherence'] = cohs

    return features


if __name__ == "__main__":

    # ---------------- My V1 Pipeline ------------------- #
    # Goal: Simulate the mapping of features from images  #
    #           Just like how humans do!                  #

    # Extra Notes: No this doesn't have a feed back loop for taking information from other Vs, It is not purely biologically designed it is more tuned to computer vision, & No it doesn't do everything the V1 system computes it rather computes enough information to prepare for V2
    # Goal Generate the raw inputs we will need to begin V2 early surface segmentation & depth estimation

    # Load any image you want from skimage.data
    img = util.img_as_float(data.chelsea())

    if img.ndim == 2:
        gray = img
        rgb = np.stack([gray]*3, axis=-1)
    else:
        rgb = img
        gray = color.rgb2gray(img)

    orientations = 8
    import time
    start=time.time()
    feat = V1_Feature_extraction(gray, rgb, fovea_radius=120, peripheral_scale=0.25, Orientations=orientations)
    print(time.time()-start, "Estimated duration >:)")

    texture = feat['texture']
    Rg = feat['R_minus_G']
    By = feat['B_minus_Y']
    scales = feat['scales']
    coherences = feat['coherence']

    plt.figure(figsize=(22,10))
    plt.subplot(1,3,1)
    plt.imshow(texture, cmap='magma')
    plt.title('Texture')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(Rg, cmap='magma')
    plt.title('RG')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(By, cmap='magma')
    plt.title('BY')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16,8))
    for i, w in enumerate(coherences):
        plt.subplot(1, len(coherences), i+1)
        plt.imshow(w, cmap='hot')
        plt.title(f"Weighted Scale σ={ [1,2,4][i] }")
        plt.axis('off')
        plt.suptitle("Coherence field / Our Edge Map", fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    for i, r in enumerate(scales):
        plt.subplot(2, 4, i+1)
        plt.imshow(r, cmap='gray')
        plt.title(f"θ = {np.round(np.pi*i/orientations, 2)} rad")
        plt.axis("off")
    plt.suptitle("Steerable Pyramid", fontsize=16)
    plt.tight_layout()
    plt.show()

