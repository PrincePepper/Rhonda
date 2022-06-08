import cv2
import numpy as np


def splitChannels(raw, pattern='RG'):
    res = np.copy(raw)

    P1 = res[0::2, 0::2]
    P2 = res[0::2, 1::2]
    P3 = res[1::2, 0::2]
    P4 = res[1::2, 1::2]

    if pattern == 'RG':
        r = P1
        gr = P2
        gb = P3
        b = P4

    return r, gr, gb, b


def joinChannels(r, gr, gb, b, pattern='RG'):
    res = np.zeros((len(r) * 2, len(r[0]) * 2))

    if pattern == 'RG':
        P1 = r
        P2 = gr
        P3 = gb
        P4 = b

    res[0::2, 0::2] = P1
    res[0::2, 1::2] = P2
    res[1::2, 0::2] = P3
    res[1::2, 1::2] = P4

    return res


class DPC:
    'Dead Pixel Correction'

    def __init__(self, img, thres, clip):
        self.img = img
        self.thres = thres
        self.clip = clip

    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        img_pad = self.padding()
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        dpc_img = np.empty((raw_h, raw_w), np.uint16)
        for y in range(img_pad.shape[0] - 4):
            for x in range(img_pad.shape[1] - 4):
                p0 = img_pad[y + 2, x + 2]
                p1 = img_pad[y, x]
                p2 = img_pad[y, x + 2]
                p3 = img_pad[y, x + 4]
                p4 = img_pad[y + 2, x]
                p5 = img_pad[y + 2, x + 4]
                p6 = img_pad[y + 4, x]
                p7 = img_pad[y + 4, x + 2]
                p8 = img_pad[y + 4, x + 4]
                if (abs(p1 - p0) > self.thres) and (abs(p2 - p0) > self.thres) and (
                        abs(p3 - p0) > self.thres) \
                        and (abs(p4 - p0) > self.thres) and (abs(p5 - p0) > self.thres) and (
                        abs(p6 - p0) > self.thres) \
                        and (abs(p7 - p0) > self.thres) and (abs(p8 - p0) > self.thres):
                    p0 = (p2 + p4 + p5 + p7) / 4
                dpc_img[y, x] = p0
        self.img = dpc_img
        return self.clipping()


class BLC:
    'Black Level Compensation'

    def __init__(self, img, parameter, clip):
        self.img = img
        self.parameter = parameter
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        bl_r = self.parameter[0]
        bl_gr = self.parameter[1]
        bl_gb = self.parameter[2]
        bl_b = self.parameter[3]
        alpha = self.parameter[4]
        beta = self.parameter[5]
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        blc_img = np.empty((raw_h, raw_w), np.int16)
        r = self.img[::2, ::2] + bl_r
        b = self.img[1::2, 1::2] + bl_b
        gr = self.img[::2, 1::2] + bl_gr + alpha * r / 256
        gb = self.img[1::2, ::2] + bl_gb + beta * b / 256
        blc_img[::2, ::2] = r
        blc_img[::2, 1::2] = gr
        blc_img[1::2, ::2] = gb
        blc_img[1::2, 1::2] = b

        self.img = blc_img
        return self.img


class CFA:
    'Color Filter Array Interpolation'

    def __init__(self, img, clip):
        self.img = img
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def malvar(self, is_color, center, y, x, img):
        if is_color == 'r':
            r = center
            g = 4 * img[y, x] - img[y - 2, x] - img[y, x - 2] - img[y + 2, x] - img[y, x + 2] \
                + 2 * (img[y + 1, x] + img[y, x + 1] + img[y - 1, x] + img[y, x - 1])
            b = 6 * img[y, x] - 3 * (
                    img[y - 2, x] + img[y, x - 2] + img[y + 2, x] + img[y, x + 2]) / 2 \
                + 2 * (img[y - 1, x - 1] + img[y - 1, x + 1] + img[y + 1, x - 1] + img[
                y + 1, x + 1])
            g = g / 8
            b = b / 8
        elif is_color == 'gr':
            r = 5 * img[y, x] - img[y, x - 2] - img[y - 1, x - 1] - img[y + 1, x - 1] - img[
                y - 1, x + 1] - img[y + 1, x + 1] - img[y, x + 2] \
                + (img[y - 2, x] + img[y + 2, x]) / 2 + 4 * (img[y, x - 1] + img[y, x + 1])
            g = center
            b = 5 * img[y, x] - img[y - 2, x] - img[y - 1, x - 1] - img[y - 1, x + 1] - img[
                y + 2, x] - img[y + 1, x - 1] - img[y + 1, x + 1] \
                + (img[y, x - 2] + img[y, x + 2]) / 2 + 4 * (img[y - 1, x] + img[y + 1, x])
            r = r / 8
            b = b / 8
        elif is_color == 'gb':
            r = 5 * img[y, x] - img[y - 2, x] - img[y - 1, x - 1] - img[y - 1, x + 1] - img[
                y + 2, x] - img[y + 1, x - 1] - img[y + 1, x + 1] \
                + (img[y, x - 2] + img[y, x + 2]) / 2 + 4 * (img[y - 1, x] + img[y + 1, x])
            g = center
            b = 5 * img[y, x] - img[y, x - 2] - img[y - 1, x - 1] - img[y + 1, x - 1] - img[
                y - 1, x + 1] - img[y + 1, x + 1] - img[y, x + 2] \
                + (img[y - 2, x] + img[y + 2, x]) / 2 + 4 * (img[y, x - 1] + img[y, x + 1])
            r = r / 8
            b = b / 8
        elif is_color == 'b':
            r = 6 * img[y, x] - 3 * (
                    img[y - 2, x] + img[y, x - 2] + img[y + 2, x] + img[y, x + 2]) / 2 \
                + 2 * (img[y - 1, x - 1] + img[y - 1, x + 1] + img[y + 1, x - 1] + img[
                y + 1, x + 1])
            g = 4 * img[y, x] - img[y - 2, x] - img[y, x - 2] - img[y + 2, x] - img[y, x + 2] \
                + 2 * (img[y + 1, x] + img[y, x + 1] + img[y - 1, x] + img[y, x - 1])
            b = center
            r = r / 8
            g = g / 8
        return [r, g, b]

    def executeB2(self):
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        img_pad = img_pad.astype(np.int32)
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)

        for y in range(0, self.img.shape[0] - 4 - 1, 2):
            for x in range(0, self.img.shape[1] - 4 - 1, 2):
                cfa_img[y, x] = [img_pad[y, x],
                                 (img_pad[y - 1, x - 1] + img_pad[y - 1, x + 1] + img_pad[
                                     y + 1, x - 1] + img_pad[
                                      y + 1, x + 1]) / 4,
                                 (img_pad[y - 1, x] + img_pad[y, x - 1] + img_pad[y, x + 1] +
                                  img_pad[y + 1, x]) / 4]

                cfa_img[y, x + 1] = [(img_pad[y, x] + img_pad[y, x + 2]) / 2,
                                     (img_pad[y - 1, x + 1] + img_pad[y + 1, y + 1]) / 2,
                                     img_pad[y, x]]

                cfa_img[y + 1, x] = [(img_pad[y, x] + img_pad[y + 2, x]) / 2,
                                     (img_pad[y + 1, x - 1] + img_pad[y + 1, y + 1] / 2),
                                     img_pad[y, x]]

                cfa_img[y + 1, x + 1] = [
                    (img_pad[y, x] + img_pad[y, x + 2] + img_pad[y + 2, x] + img_pad[
                        y + 2, x + 2]) / 2,
                    img_pad[y, x],
                    (img_pad[y - 1, x] + img_pad[y, x + 4] + img_pad[y, x - 1] + img_pad[
                        y + 1, x]) / 4]

        # for y in range(0, int(self.img.shape[0] / 2) - 4 - 1, 2):
        #     for x in range(0, int(self.img.shape[1] / 2) - 4 - 1, 2):
        #         cfa_img[y, x] = [r[y, x],
        #                          (b[y - 1, x - 1] + b[y - 1, x + 1] + b[y + 1, x - 1] + b[
        #                              y + 1, x + 1]) / 4,
        #                          (gr[y - 1, x] + gb[y, x - 1] + gr[y, x + 1] + gb[y + 1, x]) / 4]
        #
        #         cfa_img[y, x + 1] = [(r[y, x] + r[y, x + 2]) / 2,
        #                              (b[y - 1, x + 1] + b[y + 1, y + 1]) / 2,
        #                              gr[y, x]]
        #
        #         cfa_img[y + 1, x] = [(r[y, x] + r[y + 2, x]) / 2,
        #                              (b[y + 1, x - 1] + b[y + 1, y + 1] / 2),
        #                              gb[y, x]]
        #
        #         cfa_img[y + 1, x + 1] = [
        #             (r[y, x] + r[y, x + 2] + r[y + 2, x] + r[y + 2, x + 2]) / 2,
        #             b[y, x],
        #             (gb[y - 1, x] + gb[y, x + 4] + gr[y, x - 1] + gr[y + 1, x]) / 4]

        self.img = cfa_img
        return self.clipping()

    def executeB(self):
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)
        r, gr, gb, b = splitChannels(self.img)
        r_m = []
        gr_m = []
        gb_m = []
        b_m = []
        for y in range(0, int(raw_h / 2) - 1):
            a, f1, f2, c = [], [], [], []
            for x in range(0, int(raw_w / 2) - 1):
                a.append(
                    (r[y - 1, x - 1] + r[y - 1, x + 1] + r[y + 1, x - 1] + r[y + 1, x + 1]) / 4)
                f1.append((gr[y - 1, x] + gr[y, x - 1] + gr[y, x + 1] + gr[y + 1, x]) / 4)
                f2.append((gb[y - 1, x] + gb[y, x - 1] + gb[y, x + 1] + gb[y + 1, x]) / 4)
                c.append(
                    (b[y - 1, x - 1] + b[y - 1, x + 1] + b[y + 1, x - 1] + b[y + 1, x + 1]) / 4)
            r_m.append(a)
            gr_m.append(f1)
            gb_m.append(f2)
            b_m.append(c)

        cfa_img = joinChannels(r_m, gr_m, gb_m, b_m)
        self.img = cfa_img
        return self.clipping()

    def executeM(self):
        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect').astype(np.int32)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)
        for y in range(0, img_pad.shape[0] - 4 - 1, 2):
            for x in range(0, img_pad.shape[1] - 4 - 1, 2):
                b = img_pad[y + 2, x + 2]
                gb = img_pad[y + 2, x + 3]
                gr = img_pad[y + 3, x + 2]
                r = img_pad[y + 3, x + 3]

                cfa_img[y, x] = self.malvar('b', b, y + 2, x + 2, img_pad)
                cfa_img[y, x + 1] = self.malvar('gb', gb, y + 2, x + 3, img_pad)
                cfa_img[y + 1, x] = self.malvar('gr', gr, y + 3, x + 2, img_pad)
                cfa_img[y + 1, x + 1] = self.malvar('r', r, y + 3, x + 3, img_pad)

        self.img = cfa_img
        return self.img


def openRaw(path, bpp, raw_w, raw_h):
    return np.fromfile(open(path), np.dtype(bpp), raw_w * raw_h).reshape(raw_h, raw_w)


def applyBLC(raw, blcR, blcGr, blcGb, blcB):
    correctedRaw = np.zeros(raw.shape, dtype=np.int32)
    correctedRaw[0::2, 1::2] = raw[0::2, 1::2] - blcGr
    correctedRaw[0::2, 0::2] = raw[0::2, 0::2] - blcR
    correctedRaw[1::2, 0::2] = raw[1::2, 0::2] - blcGb
    correctedRaw[1::2, 1::2] = raw[1::2, 1::2] - blcB
    correctedRaw[correctedRaw < 0] = 0
    return correctedRaw.astype('u2')


def estimateBLC(raw):
    red, greenRed, greenBlue, blue = splitChannels(raw)
    rBL = np.mean(red)
    grBL = np.mean(greenRed)
    gbBL = np.mean(greenBlue)
    bBL = np.mean(blue)
    return rBL, grBL, gbBL, bBL


raw_h = 2592
raw_w = 1944

rawimg = openRaw('sfr.2592x1944p5184b14.raw', 'u2', raw_h, raw_w)
rawimg = applyBLC(rawimg, 800, 800, 800, 800)

dpc_clip = 1023  # DPC clip value
dpc_thres = 100  # DPC threshold

blc_clip = 1023  # BLC clip value
bl_r = 0  # Black level offset of Red channel
bl_gr = 0  # Black level offset of Green(R) channel
bl_gb = 0  # Black level offset of Green(B) channel
bl_b = 0  # lack level offset of Blue channel
alpha = 0  # Fusion parameter for Red channel
beta = 0  # Fusion parameter for Blue channel
bl_r, bl_gr, bl_gb, bl_b = estimateBLC(rawimg)

cfa_clip = 1023  # CFA clip value

# dead pixel correction
dpc = DPC(rawimg, dpc_thres, dpc_clip)
rawimg_dpc = dpc.execute()
print('Dead Pixel Correction Done')

# black level compensation
parameter = [bl_r, bl_gr, bl_gb, bl_b, alpha, beta]
blc = BLC(rawimg_dpc, parameter, blc_clip)
rawimg_blc = blc.execute()
print('Black Level Compensation Done')

cfa_b = CFA(rawimg, 10000)
rgbimg_cfa = cfa_b.executeB2()
print('CFA BILINEAR')

cfa_m = CFA(rawimg, 10000)  # 10000
rgbimg_cfa2 = cfa_m.executeM()
print('CFA MALVAR')

scaleTo = (1280, 720)
rawScale1 = cv2.resize(rgbimg_cfa, scaleTo, interpolation=cv2.INTER_LINEAR)
rawScale2 = cv2.resize(rgbimg_cfa2, scaleTo, interpolation=cv2.INTER_LINEAR)
colour = cv2.cvtColor(rawimg, cv2.COLOR_BAYER_RG2RGB)
rawScale3 = cv2.resize(colour, scaleTo, interpolation=cv2.INTER_LINEAR)
print('SCALE')

cv2.imshow('1', rawScale1 << 2)
cv2.imshow('2', rawScale2 << 2)
cv2.imshow('3', rawScale3 << 2)

print("OK")
cv2.waitKey(0)
