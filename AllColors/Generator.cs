using System;
using System.Collections.Generic;
using System.Threading;
using System.Numerics;

using CSGL.Graphics;

namespace AllColors {
    public class Generator : IDisposable {
        public int Width { get; private set; }
        public int Height { get; private set; }
        public ColorSource ColorSource { get; private set; }
        public Color4b[] Pixels { get; private set; }

        Color4b[] scratch;
        HashSet<Vector2i> front;

        bool running;
        Thread thread;

        public Generator(int width, int height, ColorSource colorSource) {
            if (colorSource == null) throw new ArgumentNullException(nameof(colorSource));
            ColorSource = colorSource;
            Width = width;
            Height = height;
            Pixels = new Color4b[width * height];
            scratch = new Color4b[Width * Height];
            front = new HashSet<Vector2i>();

            thread = new Thread(Run);
        }

        public void Start() {
            running = true;
            thread.Start();
        }

        public bool InBounds(Vector2i v) {
            return v.X >= 0 && v.X < Width && v.Y >= 0 && v.Y < Height;
        }

        public bool IsBlack(Color4b color) {
            return color.r == 0 && color.g == 0 && color.b == 0;
        }

        public bool IsBlack(Vector2i v) {
            return IsBlack(Pixels[GetIndex(v)]);
        }

        public Vector2i GetCoord(int index) {
            int x = index % Width;
            int y = index / Width;

            return new Vector2i(x, y);
        }

        public int GetIndex(Vector2i v) {
            return v.X + (v.Y * Width);
        }

        int GetScore(Vector2i v, Color4b color) {
            Color4b nAvg = scratch[GetIndex(v)];

            int dr = nAvg.r - color.r;
            int dg = nAvg.g - color.g;
            int db = nAvg.b - color.b;

            return dr * dr + dg * dg + db * db;
        }

        void AddToFront(Vector2i v) {
            if (InBounds(v)) {
                Update(v);
                if (IsBlack(v)) {
                    front.Add(v);
                }
            }
        }

        void SetPixel(Vector2i v, Color4b color) {
            Pixels[GetIndex(v)] = color;

            AddToFront(v + new Vector2i(-1, -1));
            AddToFront(v + new Vector2i(0, -1));
            AddToFront(v + new Vector2i(1, -1));
            AddToFront(v + new Vector2i(-1, 0));
            AddToFront(v + new Vector2i(1, -0));
            AddToFront(v + new Vector2i(-1, 1));
            AddToFront(v + new Vector2i(0, 1));
            AddToFront(v + new Vector2i(1, 1));
        }

        Color4b Get(Vector2i v, ref int num) {
            if (InBounds(v) && !IsBlack(v)) {
                num++;
                return Pixels[GetIndex(v)];
            }
            return new Color4b();
        }

        void Update(Vector2i v) {
            int num = 0;
            Color4b[] neighbors = new Color4b[8];
            neighbors[0] = Get(v + new Vector2i(-1, -1), ref num);
            neighbors[1] = Get(v + new Vector2i(0, -1), ref num);
            neighbors[2] = Get(v + new Vector2i(1, -1), ref num);
            neighbors[3] = Get(v + new Vector2i(-1, 0), ref num);
            neighbors[4] = Get(v + new Vector2i(1, -0), ref num);
            neighbors[5] = Get(v + new Vector2i(-1, 1), ref num);
            neighbors[6] = Get(v + new Vector2i(0, 1), ref num);
            neighbors[7] = Get(v + new Vector2i(1, 1), ref num);

            int r = 0;
            int g = 0;
            int b = 0;
            for (int i = 0; i < 8; i++) {
                Color4b n = neighbors[i];
                r += n.r;
                g += n.g;
                b += n.b;
            }

            r /= num;
            g /= num;
            b /= num;

            var color = new Color4b(r, g, b, 0);
            scratch[GetIndex(v)] = color;
        }

        void Run() {
            var start = new Vector2i(Width / 2, Height / 2);

            //start with one pixel colored
            SetPixel(start, ColorSource.Get());

            while (running && ColorSource.Count > 0 && front.Count > 0) {
                Color4b color = ColorSource.Get();

                int bestScore = int.MaxValue;
                Vector2i bestPos = new Vector2i();

                foreach (var v in front) {
                    int score = GetScore(v, color);
                    if (score < bestScore) {
                        bestScore = score;
                        bestPos = v;
                    }
                }

                front.Remove(bestPos);

                SetPixel(bestPos, color);
            }
        }

        public void Dispose() {
            running = false;
        }
    }
}
