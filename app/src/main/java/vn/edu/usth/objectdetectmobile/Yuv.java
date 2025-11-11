package vn.edu.usth.objectdetectmobile;

import androidx.camera.core.ImageProxy;
import java.nio.ByteBuffer;

// Simple YUV_420_888 -> ARGB8888 converter (CPU). Adequate for prototyping.
public final class Yuv {
    public static int[] toArgb(ImageProxy image){
        final int w = image.getWidth(), h = image.getHeight();
        int[] out = new int[w*h];

        ByteBuffer yb = image.getPlanes()[0].getBuffer();
        ByteBuffer ub = image.getPlanes()[1].getBuffer();
        ByteBuffer vb = image.getPlanes()[2].getBuffer();
        int yRowStride = image.getPlanes()[0].getRowStride();
        int uvRowStride = image.getPlanes()[1].getRowStride();
        int uvPixelStride = image.getPlanes()[1].getPixelStride();

        byte[] y = new byte[yb.remaining()]; yb.get(y);
        byte[] u = new byte[ub.remaining()]; ub.get(u);
        byte[] v = new byte[vb.remaining()]; vb.get(v);

        for (int j=0;j<h;j++){
            int pY = j*yRowStride;
            int pUV = (j/2)*uvRowStride;
            for (int i=0;i<w;i++){
                int Y = y[pY + i] & 0xFF;
                int U = u[pUV + (i/2)*uvPixelStride] & 0xFF;
                int V = v[pUV + (i/2)*uvPixelStride] & 0xFF;

                int C = Y - 16; int D = U - 128; int E = V - 128;
                int R = clamp((298*C + 409*E + 128)>>8);
                int G = clamp((298*C - 100*D - 208*E + 128)>>8);
                int B = clamp((298*C + 516*D + 128)>>8);
                out[j*w + i] = 0xFF000000 | (R<<16) | (G<<8) | B;
            }
        }
        return out;
    }

    public static int[] rotate(int[] src, int w, int h, int rotation){
        if (rotation % 360 == 0) return src;
        int[] dst = new int[src.length];
        switch(rotation){
            case 90:
                for (int y=0;y<h;y++){
                    for (int x=0;x<w;x++){
                        dst[x*h + (h-1-y)] = src[y*w + x];
                    }
                }
                break;
            case 180:
                for (int y=0;y<h;y++){
                    for (int x=0;x<w;x++){
                        dst[(h-1-y)*w + (w-1-x)] = src[y*w + x];
                    }
                }
                break;
            case 270:
                for (int y=0;y<h;y++){
                    for (int x=0;x<w;x++){
                        dst[(w-1-x)*h + y] = src[y*w + x];
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported rotation: " + rotation);
        }
        return dst;
    }
    private static int clamp(int v){ return v<0?0:(v>255?255:v); }
}
