# Seguridad LixySwarm

## Secretos

- Nunca guardar passwords, tokens, claves SSH o API keys dentro del repo.
- Usar variables de entorno o un `.env` local ignorado por Git.
- Rotar cualquier secreto que haya aparecido en chat, logs, shell history o commits.
- No imprimir secretos completos en logs; usar solo estado (`configured/not configured`).

## Variables Sensibles

- `LIXYSWARM_PUBLISH_TOKEN`: autoriza `POST /swarm/publish` en el relay/API.
- `LIXYSWARM_MATRIARCA_KEY`: clave de cifrado para Personal Matriarca.

Para generar una clave AES-256-GCM compatible:

```bash
python3 - <<'PY'
import base64, os
print(base64.urlsafe_b64encode(os.urandom(32)).decode())
PY
```

## Matriarca Personal

- Si `LIXYSWARM_MATRIARCA_KEY` está configurado, la memoria personal se guarda con AES-256-GCM.
- La memoria global no se cifra por diseño: es el banco compartible por `GOSSIP_DELTA`.
- Si se activa cifrado sobre una memoria personal existente en plaintext, se migra al siguiente guardado.

