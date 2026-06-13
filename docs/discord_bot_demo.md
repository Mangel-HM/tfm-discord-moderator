# Demo del bot de Discord con LoRA

Esta guia resume como ejecutar el bot en un servidor privado de Discord para una demo
observacional. El bot solo recomienda revision a moderadores humanos: no borra mensajes,
no sanciona usuarios y no ejecuta acciones automaticas.

## Configurar Discord

1. Crea una aplicacion en el Discord Developer Portal y anade un bot.
2. Activa el intent privilegiado de contenido de mensajes.
3. Invita el bot al servidor privado con permisos para leer mensajes y enviar mensajes al
   canal de moderacion.
4. Copia el token solo a `.env`. No lo guardes en Git ni en documentos.

## Variables `.env`

Para la demo LoRA, configura:

```dotenv
MODERATION_BACKEND=lora
LORA_MODEL_NAME_OR_PATH=Qwen/Qwen3.5-2B
LORA_ADAPTER_DIR=outputs/lora_jigsaw_balanced_v2
LORA_BF16=true
LORA_FP16=false
LORA_MAX_NEW_TOKENS=128
LORA_TEMPERATURE=0.0
MAX_CONTEXT_MESSAGES=6
DISCORD_TOKEN=...
DISCORD_MOD_CHANNEL_ID=...
AUTO_DELETE=false
```

`DISCORD_MOD_CHANNEL_ID` es opcional. Si no esta configurado, el bot sigue funcionando y
muestra los avisos solo por consola.

## Arrancar la demo

Ejecuta:

```powershell
just bot
```

En modo `lora`, el bot carga una sola vez el modelo base y el adapter local al arrancar.
No hace falta levantar `llama.cpp` para esta demo. El backend `baseline` sigue disponible
con `LlamaCppClient`.

## Salida interna y salida humana

El bot mantiene IDs internos estables para clasificacion y evaluacion:

- `topic`: `gaming`, `soporte`, `social_general`, `otro`;
- `risk_labels`: `sin_riesgo`, `insulto_toxicidad`, `odio_discriminacion`,
  `amenaza_violencia`, `sexual_nsfw`;
- `action`: `allow`, `review`, `warn_candidate`, `delete_candidate`.

Tanto `baseline` como `lora` se adaptan dentro del bot a la misma estructura conceptual:
`topic`, `risk_labels` y `action`. El baseline puede anadir tambien `confidence` y
`rationale` para la salida humana; LoRA puede seguir devolviendo solo los tres campos
principales.

El canal de moderacion y la consola muestran nombres legibles en ingles, por ejemplo
`Insult/toxicity`, `Other` o `Review`. Este mapeo es solo una capa de presentacion: no
cambia datasets JSONL, predicciones JSONL, metricas, prompts SFT/LoRA, validacion de
schemas ni evaluacion experimental. Tampoco requiere reentrenar LoRA.

## Que se ve durante la demo

El bot escucha mensajes normales del servidor, ignora bots y mensajes vacios, mantiene los
ultimos `MAX_CONTEXT_MESSAGES` mensajes por canal y clasifica el mensaje actual usando el
contexto solo como apoyo.

Cuando la accion sugerida no es `allow`, el canal de moderacion recibe un aviso con:

- canal y autor;
- texto del mensaje;
- cantidad de mensajes previos usados como contexto;
- accion sugerida;
- etiquetas de riesgo;
- tema;
- latencia aproximada.

Las etiquetas visibles en ese aviso son human-readable en ingles. Los IDs internos se
conservan para la logica del bot y para cualquier flujo experimental.

La demo LoRA esta alineada con el entrenamiento y evaluacion en ingles sobre Jigsaw. Los
mensajes en otros idiomas pueden procesarse tecnicamente, pero quedan fuera de la calidad
medida para el adapter actual.
