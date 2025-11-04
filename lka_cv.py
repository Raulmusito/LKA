import cv2
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Initial variables 
image_folder = "/home/raul/Desktop/master/elte/ifros_lab/lka/archive/tusimple_preprocessed/training/frames/2"
start_frame_name = "0313-2_2040.jpg"
saturation_threshold = 240       # threshold for S channel mask 
sobel_threshold = (180, 200)      # min/max thresholds for Sobel gradient
first_time = 1
left_counter = 0
right_counter = 0

# Open images, one by one from the specified directory
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# Sort the image so that they follow the correct order 
def extract_frame_number(path):
    filename = os.path.basename(path)
    match = re.search(r"_(\d+)", filename)
    return int(match.group(1)) if match else -1

def inv_homografy(p, M):

    # Convert points to homogeneous coordinates
    pt1_ipm = np.array([p[0], p[1], 1])

    # Apply inverse transform
    pt1_orig = M.dot(pt1_ipm)

    # Convert back to Cartesian coordinates
    pt1_orig /= pt1_orig[2]

    # Get integer pixel coordinates
    return tuple(pt1_orig[:2].astype(int))

def draw_text_box(img, text):


    # ====== CONSTANTS ======
    POS = (10, 10)                  # top-left corner of box
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)    # white
    BOX_COLOR = (0, 255, 0)         # green
    ALPHA = 0.4                     # transparency
    PADDING = 8                     # px padding around text
    # ========================

    # Copy image for overlay
    overlay = img.copy()

    # Compute text size
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    # Define box coordinates
    x, y = POS
    box_start = (x, y)
    box_end = (x + text_w + 2 * PADDING, y + text_h + 2 * PADDING)

    # Draw filled rectangle on overlay
    cv2.rectangle(overlay, box_start, box_end, BOX_COLOR, -1)

    # Blend overlay with transparency
    img = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

    # Draw text (non-transparent)
    text_pos = (x + PADDING, y + text_h + PADDING)
    cv2.putText(img, text, text_pos, FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return img


def fit_lane_line_from_peak(ipm_img, peak_col, window=30, min_rows=30, thresh=127, eq_degree=1):
    """
    Fit a line (x = m*y + b) around a histogram peak to estimate lane inclination.

    Inputs:
      ipm_img   : 2D image (binary or grayscale) — bird's-eye IPM image
      peak_col  : integer, column index of main histogram peak (center)
      window    : half-width in pixels around peak_col to look for lane pixels
      min_rows  : minimum number of (y,x) points required to perform fit
      thresh    : if ipm_img is grayscale, pixels > thresh are considered lane (default 127)

    Returns:
      dict with keys:
        'm'         : slope (dx/dy) of fitted line (float)
        'b'         : intercept (x = m*y + b)
        'angle_deg' : angle in degrees relative to vertical (positive -> leans right as y increases)
        'x0,y0'     : line endpoint at y=0 (float x0 may be non-integer)
        'x1,y1'     : line endpoint at y=h-1
        'valid'     : True if fit was performed, False if fallback used
        'points'    : (y_coords, x_coords) arrays used in fit
    """
    # Ensure grayscale 2D
    if ipm_img.ndim == 3:
        img = cv2.cvtColor(ipm_img, cv2.COLOR_BGR2GRAY)
    else:
        img = ipm_img.copy()

    h, w = img.shape
    x_min = max(0, peak_col - window)
    x_max = min(w, peak_col + window + 1)  # python slice exclusive

    # Binary mask of candidate lane pixels
    _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    ys = []
    xs = []

    # For each row, find bright pixels in [x_min:x_max)
    for y in range(h):
        row = mask[y, x_min:x_max]
        # get indices of non-zero pixels
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            continue
        # convert local indices to global x
        x_vals = nz + x_min
        # robust per-row position: median (less sensitive to spurious pixels)
        x_row = int(np.median(x_vals))
        ys.append(y)
        xs.append(x_row)

    ys = np.array(ys, dtype=float)
    xs = np.array(xs, dtype=float)

    result = {'coeffcients': None, 'm': None, 'b': None, 'angle_deg': None,
              'x0': None, 'y0': 0, 'x1': None, 'y1': h-1,
              'valid': False, 'points': (ys, xs), 'mse': np.float64}

    # If enough points, fit x = m*y + b
    if ys.size >= min_rows:
        # Use linear fit (x as function of y) - stable for vertical-ish lines
        coef = np.polyfit(ys, xs, eq_degree)
        error = fit_error(xs, ys, coef)
        #print (type(error['mse']))

        if eq_degree == 1:
            m,b = coef 
            x0 = m * 0 + b
            x1 = m * (h - 1) + b
            angle_rad = np.arctan(m)  # angle relative to vertical
            angle_deg = np.degrees(angle_rad)

            result.update({
                'coefficients': coef,
                'm': m, 'b': b,
                'angle_deg': angle_deg,
                'x0': x0, 'x1': x1,
                'valid': True,
                'mse': error['mse']
            })
            return result
        else:


            result.update({
                'coefficients': coef,
                'm': None, 'b': None,
                'angle_deg': None,
                'x0': None, 'x1': None,
                'valid': True,
                'mse': error['mse']
            })
            return result

    # Fallback: not enough data -> return vertical line at peak_col
    result.update({
        'm': 0.0, 'b': float(peak_col),
        'angle_deg': 0.0,
        'x0': float(peak_col), 'x1': float(peak_col),
        'valid': False
    })
    return result

def fit_error(x, y, coeffs):
    """
    Compute fit error metrics for a polynomial model.

    Args:
        x (array-like): input x values (independent variable)
        y (array-like): observed y values (dependent variable)
        coeffs (array-like): polynomial coefficients from np.polyfit

    Returns:
        dict with:
            - 'y_pred': predicted values
            - 'residuals': y - y_pred
            - 'mae': mean absolute error
            - 'rmse': root mean square error
            - 'r2': coefficient of determination
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)

    # Predicted values from polynomial
    y_pred = np.polyval(coeffs, x)

    # Residuals
    residuals = y - y_pred

    # Error metrics
    mae = np.mean(np.abs(residuals))
    mse = np.sqrt(np.mean(residuals**2))

    # R coefficient of determination
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2
    }

image_files.sort(key=extract_frame_number)

# try to find the correct image to begin the extraction
try:
    start_index = next(i for i, path in enumerate(image_files)
                       if os.path.basename(path) == start_frame_name)
except StopIteration:
    print(f"Start frame '{start_frame_name}' not found. Starting from first frame.")
    start_index = 0

i = 1
# Process each image
for img_path in image_files[start_index+0:start_index+350]: # add 120 -> line change | 185 -> error with car | 240 -> line jump
#for img_path in image_files[0:350]:   
    i += 1
    color = (0,255,0)
    #print (i)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading {img_path}")
        continue

    # Crop bottom 1/2
    height, width = frame.shape[:2] # get height and width
    cropped = frame[int(height * (1/2)):height, 0:width] # only take the lower part

    # Convert to HLS
    hls = cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # S (saturation)
    _, s_mask = cv2.threshold(s, 250, 255, cv2.THRESH_BINARY) #250

    filterd_l = cv2.inRange(l, 200, 255) #200
    sobelx_3 = cv2.Sobel(filterd_l, cv2.CV_64F, 1, 0, ksize=3)  # derivative x
    abs_sobelx_3 = np.absolute(sobelx_3) # make all the gradients positive, normaly left gradients are negative.
    scaled_sobel_3 = np.uint8(255 * abs_sobelx_3 / np.max(abs_sobelx_3)) # normalize the image


    # Threshold Sobel
    sobel_mask = cv2.inRange(scaled_sobel_3, 190, 255) #190

    # --- Combine masks ---
    combined_mask = cv2.bitwise_or(s_mask, sobel_mask)


    # IPM
    src = np.float32([
    [460, 50],  # top left
    [820, 50],  # top right
    [1040, 300], # bottom right
    [240, 300]   # bottom left
    ])
    """     dst = np.float32([
    [240, 50],  # top left
    [1040, 50],  # top right
    [1040, 300], # bottom right
    [240, 300]   # bottom left
    ]) """
    dst = np.float32([
    [350, 50],  # top left
    [930, 50],  # top right
    [930, 300], # bottom right
    [350, 300]   # bottom left
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    invM = np.linalg.inv(M)

    ipm_image = cv2.warpPerspective(combined_mask, M, (width, int(height/2)))

    col_hist = np.sum(ipm_image, axis=0)
    col_hist = np.concatenate((col_hist, [0]))


    peaks, _ = find_peaks(col_hist, distance=200)
    #print ("peaks: ",peaks)

    if len(peaks) == 2:
        left_max = peaks[0]
        right_max = peaks[1]
    elif len(peaks) > 2:
        
        peak_values = col_hist[peaks] # get the values at peaks
        sorted_indices = np.argsort(peak_values)[::-1] # sort by height (descending)
        top2_peaks = peaks[sorted_indices[:2]]   # keep the two highest
        left_max = min(top2_peaks)
        right_max = max(top2_peaks)
    
    else:
        print ("zeros!!!!!!!!!!!!!!!!")
        left_max = prev_left_max
        right_max = prev_right_max
        color = (0,0,255)
    if first_time == 0:
        #print ("prev: ", prev_left_max,"act: ", left_max)

        #print ("current", left_max, right_max)
        #print ("previous", prev_left_max, prev_right_max)
        if abs(left_max-prev_left_max) > 550:
            if left_counter > 3:
                left_counter = 0
                print ("reseted left")
            else:print ("prev: ", prev_left_max,"act: ", left_max); left_max =  prev_left_max; left_counter += 1; color = (0,0,255); print ("L")
        if abs(right_max-prev_right_max) > 550:
            if right_counter > 3:
                right_counter = 0
                print ("reseted right")
            else: print ("prev: ", prev_right_max,"act: ", right_max); right_max =  prev_right_max; right_counter += 1; color = (0,0,255); print ("R")
    
    prev_left_max = left_max
    prev_right_max = right_max
    first_time = 0


    left_line = fit_lane_line_from_peak(ipm_image, left_max, eq_degree=1)
    right_line = fit_lane_line_from_peak(ipm_image, right_max, eq_degree=1)

    # Points in IPM (bird's-eye) view
    """ LPH_1 = (left_max, 50) # left point homogray
    LPH_2 = (left_max, 300)
    RPH_1 = (right_max, 50) # left point homogray
    RPH_2 = (right_max, 300) """

    LPH_1 = (left_line["x0"], left_line["y0"]) # left point homogray
    LPH_2 = (left_line["x1"], left_line["y1"])
    RPH_1 = (right_line["x0"], right_line["y0"]) # left point homogray
    RPH_2 = (right_line["x1"], right_line["y1"]) 



    LP_1 = inv_homografy(LPH_1, invM)
    LP_2 = inv_homografy(LPH_2, invM)
    RP_1 = inv_homografy(RPH_1, invM)
    RP_2 = inv_homografy(RPH_2, invM)

    
    line_max_hist = cv2.line(cropped, LP_1, LP_2, color=color, thickness=5, lineType=cv2.LINE_AA)
    line_max_hist = cv2.line(cropped, RP_1, RP_2, color=color, thickness=5, lineType=cv2.LINE_AA)

    points = np.array([[LP_1[0],LP_1[1]],[RP_1[0],RP_1[1]],[RP_2[0],RP_2[1]],[LP_2[0],LP_2[1]]])

    img = cropped.copy()

    cv2.fillPoly(img, pts=[points], color=(255, 0, 0))
    cropped = cv2.addWeighted(cropped, 0.7, img, .3, 0)

    # !!!!!!!!!!!!!!!!!!!!!!!! DISPLAY !!!!!!!!!!!!!!!!!!!!!!!!
    white_line = np.full((1, 1280), 255, dtype=np.uint8)
    img_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
    img_rgb2 = cv2.cvtColor(ipm_image, cv2.COLOR_GRAY2RGB)
    

    #display = cv2.vconcat([ cropped,img_rgb,img_rgb2])
    cropped_up = frame[0:int(height * (1/2)), 0:width]
    display = cv2.vconcat([cropped_up, cropped ])
    left_error = str(left_line['mse'])
    right_error = str(right_line['mse'])
    text_2_display = f"LEFT MSE: {left_error} RIGHT MSE: {right_error}"

    display = draw_text_box(display, text_2_display )
    cv2.imshow("Cropped | S Mask\nSobel X | Combined", display)
    
    """ plt.plot(col_hist)
    for i in peaks:
       plt.axvline(x=i, color='red', linewidth=2, linestyle='--')
    plt.title('Column-wise Histogram')
    plt.xlabel('Column index')
    plt.ylabel('Sum of pixel intensities')
    plt.show() """
    
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
