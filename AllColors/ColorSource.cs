using System;
using System.Collections.Generic;

using CSGL.Graphics;

namespace AllColors {
    public class ColorSource {
        Queue<Color4b> queue;
        int bitDepth;

        public ColorSource(int width, int height, int seed) {            
            var list = GenColors(width, height);
            var rand = new Random(seed);
            Shuffle(list, rand);
            queue = new Queue<Color4b>(list);
        }

        List<Color4b> GenColors(int width, int height) {
            int total = width * height;
            bitDepth = GetBitDepth(total);
            List<Color4b> list = new List<Color4b>();
            int max = (int)Math.Pow(2, bitDepth);
            for (int r = 0; r < max; r++) {
                for (int g = 0; g < max; g++) {
                    for (int b = 0; b < max; b++) {
                        list.Add(new Color4b(Map(r), Map(g), Map(b), 0));
                    }
                }
            }

            return list;
        }

        int GetBitDepth(int total) {
            int bitDepth = 1;

            while ((int)Math.Pow(2, 3 * bitDepth) < total) {
                bitDepth++;
            }

            return bitDepth;
        }

        int Map(int num) {
            return ((num + 1) << (8 - bitDepth)) - 1;
        }

        void Shuffle<T>(List<T> list, Random rand) {
            int n = list.Count;
            while (n > 1) {
                n--;
                int k = rand.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public Color4b Get() {
            return queue.Dequeue();
        }

        public int Count {
            get {
                return queue.Count;
            }
        }
    }
}
