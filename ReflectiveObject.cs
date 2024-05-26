using UnityEngine;

public class ReflectiveObject : MonoBehaviour
{
    private ReflectionProbeInterpolation reflectionManager;
    private Renderer objectRenderer;
    private Material materialInstance;

    void Start()
    {
        GameObject reflectionManagerObject = GameObject.Find("ReflectionManager");
        if (reflectionManagerObject == null)
        {
            Debug.LogError("ReflectionManager GameObject not found.");
            return;
        }

        reflectionManager = reflectionManagerObject.GetComponent<ReflectionProbeInterpolation>();
        if (reflectionManager == null)
        {
            Debug.LogError("ReflectionProbeInterpolation component not found on ReflectionManager.");
            return;
        }

        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer == null)
        {
            Debug.LogError("No Renderer component found on this GameObject.");
            return;
        }
        else
        {
            materialInstance = objectRenderer.material;
        }
    }

    void Update()
    {
        if (reflectionManager != null && objectRenderer != null)
        {
            Vector3 position = transform.position;
            Texture2D reflection = reflectionManager.InterpolateReflection(position);

            // Apply the reflection texture to the object's material
            if (reflection != null)
            {
                materialInstance.SetTexture("_MainTex", reflection); // For testing purposes
                materialInstance.SetTexture("_ReflectionTex", reflection); // Use _ReflectionTex for actual reflection
            }
            else
            {
                Debug.LogError("Reflection texture is null.");
            }
        }
        else
        {
            Debug.LogError("ReflectionManager or Renderer is null.");
        }
    }
}
