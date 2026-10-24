# Guía del usuario

Esta aplicación le ayuda a obtener la **etiqueta sugerida** de una seña aislada
de la Lengua de Señas Colombiana (LSC). No traduce conversaciones ni compone
oraciones: reconoce **una seña por intento** dentro de un vocabulario de diez
palabras.

No necesita instalar nada ni usar la terminal. Solo abra la dirección que le
indique el operador en su navegador (por ejemplo `http://localhost:8000`).

## Vocabulario admitido

Hola · Buenos · Días · Tardes · Noches · Yo · Nombre · Años · Gustar · Licor

La pantalla muestra siempre esta lista. Si realiza una seña fuera de ella, el
sistema igual intentará sugerir la más parecida; una sugerencia **no es una
verificación de significado**.

## Paso a paso

1. **Activar la cámara.** Pulse “Activar cámara” y acepte el permiso del
   navegador. Si lo rechaza, podrá volver a intentarlo.
2. **Encuadrar.** Colóquese de modo que se vean su **rostro, hombros y manos**.
   La vista previa aparece como un espejo para facilitar el encuadre.
3. **Capturar.** Pulse “Capturar seña”. Verá una cuenta regresiva de
   preparación de dos segundos.
4. **Ejecutar la seña.** Durante la ventana de captura (un segundo), realice la
   seña completa, de principio a fin.
5. **Leer el resultado.** El sistema mostrará “Seña sugerida” con una etiqueta.
6. **Confirmar, corregir o repetir.**
   - **Confirmar**: indica que la sugerencia le parece correcta. No cambia el
     resultado ni se usa para entrenar.
   - **Corregir**: abre una lista para elegir la seña correcta, o “Ninguna de
     estas”. La corrección se guarda aparte, sin alterar la sugerencia original.
   - **Repetir**: inicia un nuevo intento.

## Cuando se pide repetir

A veces la aplicación responde “Repita el intento” en lugar de una sugerencia.
Esto ocurre cuando la captura no tuvo calidad suficiente (por ejemplo, no se
vieron ambas manos, había más de una persona, o el sistema no está seguro).
Siga el mensaje: ajuste el encuadre y vuelva a capturar.

## Historial

En el lateral verá hasta los **20 intentos** más recientes de esta sesión, sin
imágenes. Puede pulsar “Limpiar historial” para borrarlos. Al **recargar** la
página el historial también se borra; nada se guarda entre sesiones.

## Privacidad

La aplicación procesa sus imágenes de forma temporal y **no las guarda**. No se
envían capturas ni datos a servicios externos.

## Problemas frecuentes

| Mensaje | Qué hacer |
| --- | --- |
| Cámara denegada | Revise el permiso del navegador y pulse activar de nuevo. |
| Cámara desconectada | Reconecte la cámara y actívela otra vez. |
| Varias personas | Deje una sola persona en el encuadre. |
| Muestre ambas manos | Asegúrese de que se vean las dos manos y repita. |
| Servicio ocupado | Espere un momento y vuelva a intentar. |
| El servicio no está disponible | Avise al operador. |

Si el navegador indica que **no es compatible**, pida al operador el navegador
certificado (uno moderno con soporte de `requestVideoFrameCallback`).
