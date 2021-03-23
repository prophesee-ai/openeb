/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

package prophesee.metavision.viewer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

public class CameraFrameView extends View {


    static final int DEFAULT_WIDTH = 304;
    static final int DEFAULT_HEIGHT = 240;
    int screenW;
    int screenH;
    BitmapUpdater bitmap_updater_;
    ImageUpdater img_updater_;
    Paint paint_;
    Rect rect_dest_;

    public CameraFrameView(Context context) {
        super(context);
        bitmap_updater_ = null;
        init_geom_dest_rect(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        init(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }


    public CameraFrameView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init_geom_dest_rect(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        init(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    public CameraFrameView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init_geom_dest_rect(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        init(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    void init(int width, int height) {
        Log.d("Init", "w h " + Integer.toString(width) + " " + Integer.toString(height));

        img_updater_ = new ImageUpdater(width, height);

        paint_ = new Paint();
    }

    @Override
    public void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        screenW = w;
        screenH = h;
        double rx = ((double) w) / DEFAULT_WIDTH;
        double ry = ((double) h) / DEFAULT_HEIGHT;
        double r = rx;
        if (ry < rx) r = ry;

        //your desired sizes, converted from pixels to setMeasuredDimension's unit
        int desiredWSpec = MeasureSpec.makeMeasureSpec((int) ((double) (DEFAULT_WIDTH) * r), MeasureSpec.AT_MOST);
        int desiredHSpec = MeasureSpec.makeMeasureSpec((int) ((double) (DEFAULT_HEIGHT) * r), MeasureSpec.AT_MOST);


        this.setMeasuredDimension(desiredWSpec, desiredHSpec);
        init_geom_dest_rect(MeasureSpec.getSize(desiredWSpec), MeasureSpec.getSize(desiredHSpec));
        Log.d("TOnsizechg ", "w h " + Integer.toString(w) + " " + Integer.toString(h));

    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);

        int parentWidth = MeasureSpec.getSize(widthMeasureSpec); // receives parents width in pixel
        int parentHeight = MeasureSpec.getSize(heightMeasureSpec);


        double rx = ((double) parentWidth) / DEFAULT_WIDTH;
        double ry = ((double) parentHeight) / DEFAULT_HEIGHT;
        double r = rx;
        if (ry < rx) r = ry;

        //your desired sizes, converted from pixels to setMeasuredDimension's unit
        int desiredWSpec = MeasureSpec.makeMeasureSpec((int) ((double) (DEFAULT_WIDTH) * r), MeasureSpec.AT_MOST);
        int desiredHSpec = MeasureSpec.makeMeasureSpec((int) ((double) (DEFAULT_HEIGHT) * r), MeasureSpec.AT_MOST);


        this.setMeasuredDimension(desiredWSpec, desiredHSpec);
//        rect_dest_ = new Rect(0, 0, MeasureSpec.getSize(desiredWSpec), MeasureSpec.getSize(desiredHSpec));
        init_geom_dest_rect(MeasureSpec.getSize(desiredWSpec), MeasureSpec.getSize(desiredHSpec));
        Log.d("TDV ", "w h " + Integer.toString(MeasureSpec.getSize(parentWidth)) + " " + Integer.toString(MeasureSpec.getSize(parentHeight)));

    }

    @Override
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap_updater_ != null && img_updater_ != null) {
            bitmap_updater_.update(img_updater_.getBitmap());
        }
        img_updater_.getBitmap().prepareToDraw();
        canvas.save();
        canvas.drawBitmap(img_updater_.getBitmap(), img_updater_.get_rect_orig(), get_rect_dest(), null);
        canvas.restore();

        invalidate();
    }

    Rect get_rect_dest() {
        return rect_dest_;
    }

    public void init_geom_dest_rect(int width, int height) {
        rect_dest_ = new Rect(0, 0, width, height);

    }

    public void setBitmapUpdater(BitmapUpdater bitmap_updater) {
        bitmap_updater_ = bitmap_updater;
    }
    public interface BitmapUpdater {
        public void update(Bitmap bitmap);
    }

    static class ImageUpdater {
        Rect rect_orig_;
        private Bitmap img_;
        private int color_draw_;
        private Canvas canvas_;
        private Paint paint_;
        private int width_;

        ImageUpdater(int width, int height) {
            init_geom(width, height);
        }

        public void init_geom(int width, int height) {
            rect_orig_ = new Rect(0, 0, width, height);

            img_ = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < height; ++j) {
                    img_.setPixel(i, j, 0xFF12b4f2); // chronocam blue // j | 0xFF888888);
                }
            }
            canvas_ = new Canvas(img_);
            paint_ = new Paint();
        }

        public void setRGB(int x, int y, int c) {

            if (img_.getWidth() < x) return;
            if (img_.getHeight() < y) return;

            img_.setPixel(x, y, c);
        }

        public void clear(int c) {
            canvas_.drawColor(Color.GRAY);
        }

        public void setDrawColor(int c) {
            color_draw_ = c;
            paint_.setColor(color_draw_);
        }

        public void setWidth(int w) {
            width_ = w;
            paint_.setStrokeWidth(width_);
        }

        public void drawLine(int x1, int y1, int x2, int y2) {
            canvas_.drawLine(x1, y1, x2, y2, paint_);
        }

        public void drawElispe(int cen_x, int cen_y, int width, int height, double rot) {
            canvas_.save();
            paint_.setStyle(Paint.Style.STROKE);
            canvas_.translate(cen_x, cen_y);
            canvas_.rotate((float) -rot);
            canvas_.drawOval(new RectF(0 - width / 2, 0 - height / 2, width, height), paint_);
            paint_.setStyle(Paint.Style.FILL);
            canvas_.restore();
        }

        public Bitmap getBitmap() {
            return img_;
        }

        public void setGeom(int w, int h) {
            try {
                if (w != img_.getWidth() || h != img_.getHeight()) {
                    init_geom(w, h);
                }
            } catch (Exception e) {
                System.err.println(e);
            }


        }

        Rect get_rect_orig() {
            return rect_orig_;
        }
    }

}

