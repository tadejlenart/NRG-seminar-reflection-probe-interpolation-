using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Assertions;

public class ReflectionProbeInterpolation : MonoBehaviour
{
    public NNModel modelAsset;
    private Model model;
    private IWorker worker;

    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    public Texture2D InterpolateReflection(Vector3 position)
    {
        Tensor inputTensor = null;
        Tensor outputTensor = null;
        if (worker != null)
        {
            // Create input tensor from position
            inputTensor = new Tensor(1, 3, new float[] { position.x, position.y, position.z });
            worker.Execute(inputTensor);
            outputTensor = worker.PeekOutput();
        }

        Texture2D reflectionTexture = new Texture2D(128, 128, TextureFormat.RGB24, false);
        if (outputTensor != null)
        {
            // Convert tensor to Texture2D
            Color[] pixels = new Color[128 * 128];
            for (int h = 0; h < 128; h++)
            {
                for (int w = 0; w < 128; w++)
                {
                    float r = outputTensor[0, h, w, 0];
                    float g = outputTensor[0, h, w, 1];
                    float b = outputTensor[0, h, w, 2];
                    pixels[h * 128 + w] = new Color(r, g, b);
                }
            }
            reflectionTexture.SetPixels(pixels);
            reflectionTexture.Apply();

            // Clean up
            inputTensor.Dispose();
            outputTensor.Dispose();
        }

        return reflectionTexture;
    }

    void OnDestroy()
    {
        if (worker != null) { worker.Dispose(); }
    }
}
