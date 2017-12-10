using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private void SameSizeDimensionsShapeAndLocation(ref FloatTensor tensor)
        {
            // Check if both tensors have same size
            if (tensor.Size != size)
            {
                throw new InvalidOperationException("Tensors cannot be added since they have different sizes.");
            }
            // Check if both tensors have same number of dimensions
            if (tensor.Shape.Length != shape.Length)
            {
                throw new InvalidOperationException(
                    "Tensors cannot be added since they have different number of dimensions.");
            }

            if (dataOnGpu != tensor.dataOnGpu)
            {
                throw new InvalidOperationException(String.Format("Tensors must be on same device : {0} != {1}.", dataOnGpu, tensor.dataOnGpu));
            }
            // Check if both tensors have same shapes
            for (var i = 0; i < shape.Length; i++)
            {
                if (shape[i] != tensor.Shape[i])
                {
                    throw new InvalidOperationException("Tensors cannot be added since they have different shapes.");
                }
            }
        }

        private void SwapElements(ref int[] target, int index1, int index2)
        {
            int tmp = target[index1];
            target[index1] = target[index2];
            target[index2] = tmp;
        }

        private void SwapElements(ref long[] target, int index1, int index2)
        {
            long tmp = target[index1];
            target[index1] = target[index2];
            target[index2] = tmp;
        }

        private void AssertDim(long dim, long len)
        {
            if (dim < 0 || dim >= len)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), "Must be between 0 and shape length exclusive.");
            }
        }

        private long GetDimReduceOffset(long index, long values, long stride)
        {
            return values * stride * (index / stride) + index % stride;
        }

        private void _dimForEach(
            long interations,
            long values,
            long stride,
            Action<float[], long, long> iterator
        )
        {
            MultiThread.For(interations, (i, len) =>
            {
                float[] temp = new float[values];

                long offset = GetDimReduceOffset(i, values, stride);

                for (long v = 0; v < values; v++)
                {
                    temp[v] = data[offset + v * stride];
                }

                iterator(temp, i, temp.Length);
            });
        }
    }

    public static class MultiThread
    {
        // call func on multiple threads for the range [0..len)
        public static void For(long len, Action<long, long> func)
        {
            long nCPU = Environment.ProcessorCount;

            // only use as many threads as needed (MAX array length)
            nCPU = Math.Min(nCPU, len);

            Parallel.For(0, nCPU, workerId =>
            {
                long max = len * (workerId + 1) / nCPU;
                long offset = len * workerId / nCPU;

                for (long i = offset; i < max; i++)
                {
                    func(i, len);
                }

                // shows the ranges preccessed by each thread
                // Console.WriteLine("\tnCPU {0} | workerId {1} :: {2}-{3}", nCPU, workerId, offset, max);
            });
        }

        // proccess each item of and array on multiple threads
        public static void ForEach<T>(T[] data, Action<T, long, long> func)
        {
            MultiThread.For(data.Length, (index, len) => func(data[index], index, len));
        }

        // reduce items of array on multiple threads
        public static T Reduce<T>(T[] data, Func<T, T, long, T[], T> func)
        {
            long len = data.Length;
            long nCPU = Environment.ProcessorCount;

            // only use as many threads as needed (MAX array length / 2)
            // because reduce func proccesses two items at once
            nCPU = Math.Min(nCPU, len / 2);

            T[] prev = data;

            for (; nCPU > 0; nCPU = nCPU / 2)
            {
                len = prev.Length;

                T[] curr = new T[nCPU];

                Parallel.For(0, nCPU, workerId =>
                {
                    long max = len * (workerId + 1) / nCPU;
                    long offset = len * workerId / nCPU;

                    T acc = prev[offset];

                    for (long i = offset + 1; i < max; i++)
                    {
                        acc = func(acc, prev[i], i, data);
                    }

                    curr[workerId] = acc;

                    // shows the number of items preccessed by each thread
                    // Console.WriteLine("\tnCPU {0} | workerId {1} :: {2}", nCPU, workerId, max - offset);
                });

                prev = curr;

                // show when a reduce iteration completes
                // Console.WriteLine("\t-- reduce iteration complete --");
            }

            // return the reduced value
            return prev[0];
        }
    }
}
