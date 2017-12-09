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


[Test]
public void Cos()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor.Gpu(shader);

	float[] data2 = { 0.92106099f,  0.87758256f,  0.95533649f,  0.99500417f };
	int[] shape2 = { 4 };
	var expectedCosTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedCosTensor.Gpu(shader);
	var actualCosTensor = tensor.Cos();
	actualCosTensor.Gpu(shader);

	AssertEqualTensorsData(expectedCosTensor, actualCosTensor);
}

[Test]
public void Cos_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor.Gpu(shader);
	tensor.Cos (inline: true);

	float[] data2 = {  0.92106099f,  0.87758256f,  0.95533649f,  0.99500417f };
	int[] shape2 = { 4 };
	var expectedCosTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedCosTensor.Gpu(shader);

	AssertEqualTensorsData(tensor, expectedCosTensor);
}

[Test]
public void AddScalar()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);
	float[] data2 = { -101, -100, -99.9f, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {3, 2};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);
	float scalar = -100;

	var tensor2 = tensor1.Add (scalar);

	AssertEqualTensorsData(expectedTensor, tensor2);
}

[Test]
public void AddScalar_()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { -101, -100, -99.9f, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {3, 2};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	float scalar = -100;

	tensor1.Add (scalar, inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void Add()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int[] shape1 = {2, 5};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
	int[] shape2 = {2, 5};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
	int[] shape3 = {2, 5};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	var tensorSum = tensor1.Add (tensor2);

	AssertEqualTensorsData(expectedTensor, tensorSum);
}

[Test]
public void AddUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor1.Gpu(shader);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void Add_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int[] shape1 = {2, 5};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
	int[] shape2 = {2, 5};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
	int[] shape3 = {2, 5};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	tensor1.Add (tensor2, inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void AddUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void Abs()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 0, 1, float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	var tensor2 = tensor1.Abs();
	AssertEqualTensorsData(expectedTensor, tensor2);
}

[Test]
public void Abs_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 0, 1, float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	tensor1.Abs(inline: true);
	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void Neg()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);
	float[] data2 = { 1, 0, -1, -float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	var result = tensor1.Neg ();

	AssertEqualTensorsData(expectedTensor, result);
}

[Test]
public void Sign()
{
	float[] data1 = {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
	int[] shape1 = { 1, 10 };

	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape1);
	expectedTensor.Gpu(shader);
	var result1 = tensor1.Sign();

	AssertEqualTensorsData(expectedTensor, result1);
}

[Test]
public void Sign_()
{
	float[] data1 = {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
	int[] shape1 = { 1, 10 };

	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape1);
	expectedTensor.Gpu(shader);

	tensor1.Sign (inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void Zero_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 0, 0, 0, 0, 0 };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	tensor1.Zero_ ();

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void MultiplicationElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { float.PositiveInfinity, 100, 2.25f, 0, 2.25f, 100, 400, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	var tensor3 = tensor1.Mul (tensor2);

	AssertEqualTensorsData(expectedTensor, tensor3);
}

[Test]
public void MultiplicationElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MulElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { float.PositiveInfinity, 100, 2.25f, 0, 2.25f, 100, 400, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	tensor1.Mul (tensor2, inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void MultiplicationElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);

	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwisenUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);

	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { 1, 1, 1, (float)Double.NaN, 1, 1, 1, 1 };
	int[] shape3 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	var tensor3 = tensor1.Div (tensor2);

	AssertEqualTensorsData(expectedTensor, tensor3);
}

[Test]
public void DivisionElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 1, 1, (float)Double.NaN, 1, 1, 1, 1 };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	tensor1.Div (tensor1, inline: true);

	AssertEqualTensorsData(tensor2, tensor1);
}

[Test]
public void DivisionElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationScalar()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	// Test multiplication by 0
	float scalar = 0;
	var result = tensor1.Mul (scalar);

	float[] data2 = { 0, 0, 0, 0, 0, 0, 0, 0 };
	int[] shape2 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	AssertEqualTensorsData(expectedTensor, result);

	// Test multiplication by positive
	scalar = 99;
	result = tensor1.Mul (scalar);

	float[] data3 = { float.NegativeInfinity, -990, -148.5f, 0, 148.5f, 990, 1980, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	AssertEqualTensorsData(expectedTensor, result);

	// Test multiplication by negative
	scalar = -99;
	result = tensor1.Mul (scalar);

	float[] data4 = { float.PositiveInfinity, 990, 148.5f, 0, -148.5f, -990, -1980, float.NegativeInfinity };
	int[] shape4 = {2, 4};
	expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data4, _shape: shape4);
	expectedTensor.Gpu(shader);

	AssertEqualTensorsData(expectedTensor, result);

	// Test multiplication by decimal
	scalar = 0.000001f;
	result = tensor1.Mul (scalar);

	float[] data5 = { float.MinValue * scalar, -0.000010f, -0.0000015f, 0, 0.0000015f, 0.000010f, 0.000020f, float.MaxValue * scalar};
	int[] shape5 = {2, 4};
	expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data5, _shape: shape5);
	expectedTensor.Gpu(shader);

	AssertEqualTensorsData(expectedTensor, result);
}

[Test]
public void Ceil()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 6, -20, 9, 101, 101 };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	var result = tensor1.Ceil ();

	AssertEqualTensorsData(expectedTensor, result);
}

[Test]
public void Ceil_()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 6, -20, 9, 101, 101 };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	tensor1.Ceil (inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void Floor()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 5, -21, 9, 100, 100 };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	var result = tensor1.Floor(inline: true);

	AssertEqualTensorsData(expectedTensor, result);
}

[Test]
public void Floor_()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);
	float[] data2 = { 5, -21, 9, 100, 100 };
	int[] shape2 = { 5 };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	tensor1.Floor(inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void SubtractElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { float.NegativeInfinity, -20, -3, 0, 3, 20, 40, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	var result = tensor1.Sub (tensor2);

	AssertEqualTensorsData(expectedTensor, result);
}

[Test]
public void SubtractElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	float[] data3 = { float.NegativeInfinity, -20, -3, 0, 3, 20, 40, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data3, _shape: shape3);
	expectedTensor.Gpu(shader);

	tensor1.Sub (tensor2, inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void SubtractElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	tensor2.Gpu(shader);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractScalar()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { -101, -100, -99.9f, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {3, 2};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	float scalar = 100;
	var tensor3 = tensor1.Sub (scalar);

	AssertEqualTensorsData(expectedTensor, tensor3);
}

[Test]
public void SubtractScalar_()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	tensor1.Gpu(shader);

	float[] data2 = { -101, -100, -99.9f, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {3, 2};
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: data2, _shape: shape2);
	expectedTensor.Gpu(shader);

	float scalar = 100;
	tensor1.Sub (scalar, inline: true);

	AssertEqualTensorsData(expectedTensor, tensor1);
}

[Test]
public void AddMatrixMultiplyTest()
{
	float[] base1_data = new float[] { 1, 2, 3, 4 };
	int[] base1_shape = new int[] { 2, 2 };
	var base1 = new FloatTensor(_ctrl: ctrl, _data: base1_data, _shape: base1_shape);

	float[] base2_data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] base2_shape = new int[] { 3, 3 };
	var base2 = new FloatTensor(_ctrl: ctrl, _data: base2_data,_shape: base2_shape );

	base1.Gpu(shader); base2.Gpu(shader);

	float[] data = new float[] { 1, 2, 3, 4, 5, 6 };
	int[] tensor1_shape = new int[] { 2, 3 };
	int[] tensor2_shape = new int[] { 3, 2 };

	var tensor1 = new FloatTensor(_ctrl: ctrl, _data: data, _shape: tensor1_shape);
	var tensor1Cpu = new FloatTensor(_ctrl: ctrl, _data: data, _shape: tensor1_shape);
	var tensor2 = new FloatTensor(_ctrl: ctrl, _data: data, _shape: tensor2_shape);
	var tensor2Cpu = new FloatTensor(_ctrl: ctrl, _data: data, _shape: tensor2_shape);

	tensor1.Gpu(shader); tensor2.Gpu(shader);

	base1.AddMatrixMultiply(tensor1, tensor2);
	base2.AddMatrixMultiply(tensor2, tensor1);

	float[] expectedData1 = new float[base1_shape[0] * base1_shape[1]];

	for (int i = 0; i < base1_shape[0]; i++)
	{
		for (int j = 0; j < base1_shape[1]; j++)
		{
			int expectedDataIndex = i * base1_shape[1] + j;
			expectedData1 [expectedDataIndex] = base1_data [expectedDataIndex];

			for (int k = 0; k < tensor1_shape[1]; k++)
			{
				expectedData1 [expectedDataIndex] += tensor1Cpu[i, k] * tensor2Cpu[k, j];
			}
		}
	}
	var expectedTensor1 = new FloatTensor(_ctrl: ctrl, _data: expectedData1, _shape: base1_shape);
	expectedTensor1.Gpu(shader);
	AssertEqualTensorsData(expectedTensor1, base1);

	float[] expectedData2 = new float[base2_shape[0] * base2_shape[1]];

	for (int i = 0; i < base2_shape[0]; i++)
	{
		for (int j = 0; j < base2_shape[1]; j++)
		{
			int expectedDataIndex = i * base2_shape[1] + j;
			expectedData2 [expectedDataIndex] = base2_data [expectedDataIndex];
			for (int k = 0; k < tensor2_shape[1]; k++)
			{
				expectedData2 [expectedDataIndex] += tensor2Cpu[i, k] * tensor1Cpu[k, j];
			}
		}
	}

	var expectedTensor2 = new FloatTensor(_ctrl: ctrl, _data: expectedData2, _shape: base2_shape);
	expectedTensor2.Gpu(shader);
	AssertEqualTensorsData(expectedTensor2, base2);
}

[Test]
public void AddMatrixVectorProductTest()
{
	float[] baseData = new float[] { 1, 2 };
	int[] baseShape = new int[] { 2 };
	var baseVector = new FloatTensor(_ctrl: ctrl, _data: baseData, _shape: baseShape);
	baseVector.Gpu(shader);

	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = new int[] { 2, 2 };
	var matrix = new FloatTensor(_ctrl: ctrl, _data: data1, _shape: shape1);
	matrix.Gpu(shader);

	float[] data2 = new float[] { 5, 6 };
	int[] shape2 = new int[] { 2 };
	var vector = new FloatTensor (_ctrl: ctrl, _data: data2, _shape: shape2);
	vector.Gpu(shader);

	baseVector.AddMatrixVectorProduct(matrix, vector);

	float[] expectedData = new float[] { 18, 41 };
	int[] expectedShape = new int[] { 2 };
	var expectedVector = new FloatTensor(_ctrl: ctrl, _data: expectedData, _shape: expectedShape);
	expectedVector.Gpu(shader);

	AssertEqualTensorsData(expectedVector, baseVector);
}

[Test]
public void Trunc()
{
	float[] data = { -0.323232f, 0.323893f, 0.99999f, 1.2323389f };
	int[] shape = { 4 };
	var tensor = new FloatTensor(_ctrl: ctrl, _data: data, _shape: shape);
	tensor.Gpu(shader);

	float[] truncatedData = { -0f, 0f, 0f, 1f };
	var expectedTensor = new FloatTensor(_ctrl: ctrl, _data: truncatedData, _shape: shape);
	expectedTensor.Gpu(shader);

	var truncatedTensor = tensor.Trunc();

	AssertEqualTensorsData(expectedTensor, truncatedTensor);
}



}
}
