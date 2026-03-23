#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# upload_data_to_drive.sh
#
# Sube la carpeta data/ del proyecto a Google Drive usando rclone.
# Destino: MyDrive/galaxy-morph-ml/data/
#
# Uso:
#   1. Instalar rclone:  sudo pacman -S rclone   (Manjaro/Arch)
#   2. Configurar:       rclone config
#      → Elegir "Google Drive", seguir el flow OAuth en el navegador
#      → Nombrar el remote como "gdrive" (o cambiar REMOTE abajo)
#   3. Ejecutar:         bash scripts/upload_data_to_drive.sh
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

REMOTE="gdrive"                                    # nombre del remote en rclone config
LOCAL_DATA="$(cd "$(dirname "$0")/.." && pwd)/data" # carpeta data/ del proyecto
DRIVE_DEST="${REMOTE}:galaxy-morph-ml/data"         # destino en Drive

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Upload data/ → Google Drive                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Local:  ${LOCAL_DATA}"
echo "  Remote: ${DRIVE_DEST}"
echo ""

# Verificar rclone
if ! command -v rclone &>/dev/null; then
    echo "⛔ rclone no está instalado."
    echo "   Instálalo con:  sudo pacman -S rclone"
    exit 1
fi

# Verificar que el remote existe
if ! rclone listremotes | grep -q "^${REMOTE}:$"; then
    echo "⛔ Remote '${REMOTE}' no encontrado en rclone config."
    echo "   Ejecuta:  rclone config"
    echo "   Y crea un remote llamado '${REMOTE}' para Google Drive."
    exit 1
fi

# Verificar carpeta local
if [ ! -d "${LOCAL_DATA}" ]; then
    echo "⛔ Carpeta no encontrada: ${LOCAL_DATA}"
    exit 1
fi

# Mostrar qué se va a subir
echo "📊 Contenido a subir:"
du -sh "${LOCAL_DATA}"/* 2>/dev/null || true
echo ""

# Subir con progreso
echo "📤 Subiendo... (esto puede tomar varios minutos para ~3 GB)"
echo ""
rclone copy "${LOCAL_DATA}" "${DRIVE_DEST}" \
    --progress \
    --transfers=4 \
    --checkers=8 \
    --drive-chunk-size=64M \
    --exclude=".gitkeep" \
    --exclude="processed/**"

echo ""
echo "✅ Upload complete!"
echo "   Verifica en Drive: MyDrive/galaxy-morph-ml/data/"
