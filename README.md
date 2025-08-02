
## ⚠️ Nota sobre errores libGL al ejecutar CarRacing-v3
Para tener la ventana de Gymnasium funcionando en el contenedor, primero se debe ejecutar en el host:
```bash
xhost +local:root
# Output: non-network local connections being added to access control list
```

Al ejecutar el entorno `CarRacing-v3` con `render_mode="human"`, puede aparecer el siguiente warning en la consola:

```bash
libGL error: glx: failed to create dri3 screen
libGL error: failed to load driver: nouveau
```
### 🟢 Este mensaje puede ignorarse.
No afecta la ejecución ni el renderizado del entorno.

### 📌 Explicación:
El contenedor intenta usar el driver gráfico nouveau (open-source), pero al estar configurado para usar GPU vía nvidia-container-runtime y CUDA, este driver no es necesario. El entorno sigue funcionando correctamente con los drivers de NVIDIA instalados en el sistema anfitrión.

Script de ejecución
```bash
python -m scripts.train_image_recons \
    --batch_size 64 \
    --epochs 100 \
    --run_dir runs/imageReconstructor/test1
```

Con batch_size=128 se alcanza el uso de GPU=80%


Para medir el uso de la GPU - Funciona fuera del contenedor
```bash
watch -n1 nvidia-smi
```

Para ver el tamaño del modelo:
```bash
python -m scripts.inspect_params
```