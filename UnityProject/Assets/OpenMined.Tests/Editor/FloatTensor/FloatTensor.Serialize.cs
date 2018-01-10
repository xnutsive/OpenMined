using UnityEngine;
using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using OpenMined.Syft.NN;
using OpenMined.Syft.Layer;

namespace OpenMined.Tests.Editor.FloatTensor
{
    [Category("FloatTensor Serialization Tests")]
    public class FloatTensorSerializationTests
    {
        private SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
        }

        [Test]
        public void ReserializeTest()
        {
            float[] d = { 1f, 2f, 3f, 4f, 5f };
            int[] s = { 5 };

            var tensor = ctrl.floatTensorFactory.Create(_data: d, _shape: s);
            var serialized = JsonUtility.ToJson(tensor);

            var deserializedTensor = JsonUtility.FromJson<OpenMined.Syft.Tensor.FloatTensor>(serialized);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(deserializedTensor.Data[i], tensor.Data[i]);
            }

            for (int i = 0; i < tensor.Shape.Length; i++)
            {
                Assert.AreEqual(deserializedTensor.Shape[i], tensor.Shape[i]);
            }
        }

        [Test]
        public void ReserializeTestComplex()
        {
            float[] d = { 1f, 2f, 3f, 4f, 5f, 6.5f, 7f, 8f, 9f, 10f };
            int[] s = { 2, 3 };

            var tensor = ctrl.floatTensorFactory.Create(_data: d, _shape: s);
            var serialized = JsonUtility.ToJson(tensor);

            var deserializedTensor = JsonUtility.FromJson<OpenMined.Syft.Tensor.FloatTensor>(serialized);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(deserializedTensor.Data[i], tensor.Data[i]);
            }

            for (int i = 0; i < tensor.Shape.Length; i++)
            {
                Assert.AreEqual(deserializedTensor.Shape[i], tensor.Shape[i]);
            }
        }

        [Test]
        public void SerializeModel()
        {
            var model = new Sequential(ctrl);
            model.AddLayer(new Linear(ctrl, 10, 5));
            // model.AddLayer(new ReLU(ctrl));
            // model.AddLayer(new Dropout(ctrl, 0.5f));
            model.AddLayer(new Linear(ctrl, 5, 2));
            //model.AddLayer(new Softmax(ctrl, 0));
            //model.AddLayer(new Log(ctrl));

            var serialized = JsonUtility.ToJson(model);
            Debug.Log("Yooooo ");
        }
    }
}