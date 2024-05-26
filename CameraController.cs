using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float speed = 10.0f;
    public float lookSpeed = 2.0f;
    public float lookXLimit = 45.0f;

    private float rotationX = 0;

    void Start()
    {
        // Lock the cursor
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        // Camera rotation with mouse
        rotationX += -Input.GetAxis("Mouse Y") * lookSpeed;
        rotationX = Mathf.Clamp(rotationX, -lookXLimit, lookXLimit);
        float rotationY = transform.localEulerAngles.y + Input.GetAxis("Mouse X") * lookSpeed;

        transform.localEulerAngles = new Vector3(rotationX, rotationY, 0);

        // Camera movement with keyboard
        float moveDirectionX = Input.GetAxis("Horizontal");
        float moveDirectionZ = Input.GetAxis("Vertical");
        Vector3 move = transform.right * moveDirectionX + transform.forward * moveDirectionZ;

        // Move up with Spacebar, down with Left Shift
        if (Input.GetKey(KeyCode.Space))
        {
            move += transform.up;
        }
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            move -= transform.up;
        }

        transform.position += move * speed * Time.deltaTime;
    }
}
