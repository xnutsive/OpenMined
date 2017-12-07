using System;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Network.Servers;

using OpenMined.Syft.Tensor;
using UnityEditor.VersionControl;

namespace OpenMined.Tests
{

public class FloatTensorGpuTest
{

public SyftController ctrl;
public ComputeShader shader;

public void AssertEqualTensorsData(FloatTensor t1, FloatTensor t2) {

	float[] data1 = new float[t1.Size]; t1.DataBuffer.GetData(data1);
	float[] data2 = new float[t2.Size]; t2.DataBuffer.GetData(data2);

	Assert.AreEqual(t1.DataBuffer.count, t2.DataBuffer.count);
	Assert.AreEqual(t1.DataBuffer.stride, t2.DataBuffer.stride);
	Assert.AreNotEqual(t1.DataBuffer.GetNativeBufferPtr(), t2.DataBuffer.GetNativeBufferPtr());
	Assert.AreEqual(data1, data2);
}

[TestFixtureSetUp]
public void Init()
{
	//Init runs once before running test cases.
	ctrl = new SyftController(null);
	shader = Camera.main.GetComponents<SyftServer>()[0].Shader;
}

[TestFixtureTearDown]
public void CleanUp()
{
	//CleanUp runs once after all test cases are finished.
}

[SetUp]
public void SetUp()
{
	//SetUp runs before all test cases
}

[TearDown]
public void TearDown()
{
	//SetUp runs after all test cases
}

[Test]
public void Copy()
{
	float[] array = { 1, 2, 3, 4, 5 };
	int[] shape = { 5 };

	var tensor = new FloatTensor(_ctrl: ctrl, _data: array, _shape: shape);
	tensor.Gpu(shader);
	var copy = tensor.Copy();
	copy.Gpu(shader);

	Assert.AreEqual(copy.Shape,tensor.Shape);
	Assert.AreNotEqual(copy.Id, tensor.Id);
	AssertEqualTensorsData(tensor, copy);
}

// TODO: AddMatrixMultiplyTests when implemented on CPU
// TODO: MultiplyDerivative when implemented on CPU
}
}
