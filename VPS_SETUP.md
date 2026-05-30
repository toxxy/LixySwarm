# VPS Setup — LixySwarm Primer Nodo Externo

## Qué necesita el VPS

El VPS actuará como **primer nodo externo** del enjambre. Puede:
1. Conectarse al nodo local de Emmanuel via P2P
2. Participar en el enjambre compartiendo feromonas
3. (Futuro) Contribuir capacidad de cómputo al training distribuido

---

## Paso 1 — Requisitos del VPS

```bash
# Mínimo recomendado:
# RAM: 8GB (16GB ideal para el modelo completo)
# CPU: 4 cores (8 ideal)
# Disk: 20GB libres
# OS: Ubuntu 22.04 / 24.04

# Verificar
free -h && nproc && df -h /
```

## Paso 2 — Instalar dependencias

```bash
# Python 3.11+
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clonar el repo
git clone https://github.com/toxxy/LixySwarm.git
cd LixySwarm

# Entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Dependencias (CPU-only en VPS sin GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Paso 3 — Descargar checkpoint

El VPS necesita el modelo entrenado. Opciones:

**Opción A — rsync desde local (recomendado):**
```bash
# Desde la máquina de Emmanuel:
rsync -avz --progress \
  /home/toxxy/Dropbox/Lixy/clawd/workspace/lixy-llm/checkpoints/swarm_best.pt \
  usuario@VPS_IP:/home/usuario/LixySwarm/checkpoints/
```

**Opción B — scp directo:**
```bash
scp checkpoints/swarm_best.pt usuario@VPS_IP:~/LixySwarm/checkpoints/
```

## Paso 4 — Configurar el nodo

```bash
# En el VPS, crear config de nodo
cat > node_config.json << 'EOF'
{
  "node_id": "vps-node-01",
  "listen_host": "0.0.0.0",
  "listen_port": 7337,
  "peer_nodes": [
    "EMMANUEL_IP:7337"
  ],
  "role": "relay",
  "feromon_dim": 256,
  "checkpoint": "checkpoints/swarm_best.pt"
}
EOF
```

## Paso 5 — Abrir puertos en el firewall del VPS

```bash
# UDP 7337: feromon broadcast
# TCP 7338: gossip protocol
sudo ufw allow 7337/udp
sudo ufw allow 7337/tcp
sudo ufw allow 7338/tcp
sudo ufw status
```

## Paso 6 — Levantar el nodo como servicio

```bash
# Crear systemd service
sudo tee /etc/systemd/system/lixyswarm.service << 'EOF'
[Unit]
Description=LixySwarm Node
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/LixySwarm
ExecStart=/home/ubuntu/LixySwarm/venv/bin/python lixy_orchestrator.py --node-mode
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable lixyswarm
sudo systemctl start lixyswarm
sudo systemctl status lixyswarm
```

## Paso 7 — Verificar conectividad P2P

```bash
# Desde el VPS, probar conexión al nodo local
python3 -c "
from src.network.swarm_network import SwarmNetwork
net = SwarmNetwork(host='0.0.0.0', port=7337)
net.start()
net.connect_peer('EMMANUEL_IP', 7337)
print('Conexión P2P OK')
"
```

---

## Estado actual del protocolo P2P

El `SwarmNetwork` ya implementa:
- ✅ UDP loopback feromon broadcast
- ✅ TCP gossip bidireccional
- ✅ `inject_remote_feromon()` + `merge_remote_feromons()`
- ✅ 23/23 tests pasando + 15/15 tests de integración
- ⏳ mDNS (LAN auto-discovery) — funciona en LAN, el VPS necesita IP explícita

---

## Próximos pasos después del VPS

1. VPS conectado → primer test de feromon cross-internet
2. Medir latencia de feromona (target: <100ms)
3. SwarmExplorer: el VPS puede exponer el dashboard público
4. El VPS se convierte en relay para futuros nodos de la red

---

*Preparado por Cody | 2026-05-30*
