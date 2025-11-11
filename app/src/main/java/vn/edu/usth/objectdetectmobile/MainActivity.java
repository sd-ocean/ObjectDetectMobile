package vn.edu.usth.objectdetectmobile;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

// NEW imports for the non-deprecated API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtException;

public class MainActivity extends ComponentActivity {
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    private static final long DEPTH_INTERVAL_MS = 1500L;
    private static final long DEPTH_CACHE_MS = 3000L;
    private static final boolean ENABLE_INPUT_BLUR = true;
    private static final int BLUR_RADIUS = 1; // 1 => kernel 3x3

    private PreviewView previewView;
    private OverlayView overlay;
    private ObjectDetector detector;
    private DepthEstimator depthEstimator;
    private ExecutorService exec;
    private long lastDepthMillis = 0L;
    private DepthEstimator.DepthMap lastDepthMap = null;
    private long lastDepthCacheTime = 0L;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        overlay.setLabels(loadLabels());

        exec = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            start();
        }
    }

    private void start() {
        try {
            detector = new ObjectDetector(this);
        } catch (Throwable e) {
            Log.e(TAG, "Detector init failed", e);
            Toast.makeText(this, "Detector load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return; // detector is required
        }

        try {
            depthEstimator = new DepthEstimator(this);
        } catch (Throwable e) {
            Log.w(TAG, "Depth estimator disabled", e);
            depthEstimator = null;
            lastDepthMap = null;
        }

        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(this).get();
                provider.unbindAll();

                // Build Preview with ResolutionSelector (non-deprecated)
                Preview preview =
                        new Preview.Builder()
                                .setResolutionSelector(
                                        new ResolutionSelector.Builder()
                                                .setAspectRatioStrategy(
                                                        AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                                )
                                                .build()
                                )
                                .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Build ImageAnalysis with ResolutionSelector (prefer ~640 square input, but let CameraX choose closest)
                ImageAnalysis analysis =
                        new ImageAnalysis.Builder()
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .setResolutionSelector(
                                        new ResolutionSelector.Builder()
                                                .setAspectRatioStrategy(
                                                        AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                                )
                                                .setResolutionStrategy(
                                                        new ResolutionStrategy(
                                                                new Size(360, 360),
                                                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                                        )
                                                )
                                                .build()
                                )
                                .build();

                analysis.setAnalyzer(exec, image -> {
                    try {
                        int frameW = image.getWidth();
                        int frameH = image.getHeight();
                        int rotation = image.getImageInfo().getRotationDegrees();
                        int[] argb = Yuv.toArgb(image);
                        if (rotation != 0) {
                            argb = Yuv.rotate(argb, frameW, frameH, rotation);
                            if (rotation == 90 || rotation == 270) {
                                int tmp = frameW;
                                frameW = frameH;
                                frameH = tmp;
                            }
                        }

                        int[] detectorInput = argb;
                        if (ENABLE_INPUT_BLUR && BLUR_RADIUS > 0) {
                            detectorInput = boxBlur(argb, frameW, frameH, BLUR_RADIUS);
                        }

                        // Run detection and depth off the UI thread
                        List<ObjectDetector.Detection> dets =
                                detector.detect(detectorInput, frameW, frameH);

                        if (depthEstimator != null) {
                            long now = SystemClock.elapsedRealtime();
                            boolean shouldRunDepth = now - lastDepthMillis >= DEPTH_INTERVAL_MS;
                            if (shouldRunDepth) {
                                try {
                                    DepthEstimator.DepthMap depth = depthEstimator.estimate(argb, frameW, frameH);
                                    dets = depthEstimator.attachDepth(dets, depth);
                                    lastDepthMillis = SystemClock.elapsedRealtime();
                                    lastDepthMap = depth;
                                    lastDepthCacheTime = lastDepthMillis;
                                } catch (Throwable depthErr) {
                                    Log.w(TAG, "Depth inference disabled due to error", depthErr);
                                    try {
                                        depthEstimator.close();
                                    } catch (Exception ignore) {}
                                    depthEstimator = null;
                                    lastDepthMap = null;
                                }
                            } else if (lastDepthMap != null && now - lastDepthCacheTime <= DEPTH_CACHE_MS) {
                                dets = depthEstimator.attachDepth(dets, lastDepthMap);
                            }
                        }

                        // Draw on overlay on UI thread
                        int finalW = frameW;
                        int finalH = frameH;
                        List<ObjectDetector.Detection> finalDets = dets;
                        runOnUiThread(() -> overlay.setDetections(finalDets, finalW, finalH));
                    } catch (OrtException t) {
                        Log.e(TAG, "detect failed", t);
                    } catch (Throwable t) {
                        Log.e(TAG, "analyzer crash", t);
                    } finally {
                        // ALWAYS close frame or pipeline can stall/crash
                        image.close();
                    }
                });

                CameraSelector selector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                provider.bindToLifecycle(this, selector, preview, analysis);
            } catch (Throwable e) {
                Log.e(TAG, "Camera bind error", e);
                Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private String[] loadLabels() {
        List<String> list = new ArrayList<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line;
            while ((line = br.readLine()) != null) list.add(line);
        } catch (Exception ignored) {
        } finally {
            try { if (br != null) br.close(); } catch (Exception ignored) {}
        }
        return list.toArray(new String[0]);
    }

    @Override public void onRequestPermissionsResult(int c, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(c,p,r);
        if (c == REQ && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) start();
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (exec != null) exec.shutdownNow();
        if (detector != null) {
            try {
                detector.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        if (depthEstimator != null) {
            try {
                depthEstimator.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        lastDepthMap = null;
    }

    private static int[] boxBlur(int[] src, int w, int h, int radius) {
        int kernel = (radius * 2 + 1) * (radius * 2 + 1);
        int[] dst = new int[w * h];
        for (int y = 0; y < h; y++) {
            int yMin = Math.max(0, y - radius);
            int yMax = Math.min(h - 1, y + radius);
            for (int x = 0; x < w; x++) {
                int xMin = Math.max(0, x - radius);
                int xMax = Math.min(w - 1, x + radius);

                int count = 0;
                int sumR = 0, sumG = 0, sumB = 0;
                for (int yy = yMin; yy <= yMax; yy++) {
                    int base = yy * w;
                    for (int xx = xMin; xx <= xMax; xx++) {
                        int c = src[base + xx];
                        sumR += (c >> 16) & 0xFF;
                        sumG += (c >> 8) & 0xFF;
                        sumB += c & 0xFF;
                        count++;
                    }
                }
                if (count == 0) count = 1;
                int r = sumR / count;
                int g = sumG / count;
                int b = sumB / count;
                dst[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return dst;
    }
}
