# Push this staging directory to Hugging Face Spaces

## One-time setup

1. Create a Space at https://huggingface.co/new-space
   - Name: `cmapss-rul`
   - SDK: **Docker**
   - Hardware: **CPU Basic**
   - Visibility: **Public**

2. Sync serving files (from **repo root**):

```bash
cd "/Users/alvinalias/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data Science/Resume/Portfolio Build/active/cmapss_rul"
bash deploy/hf_space/sync_to_space.sh
```

3. Clone the empty Space repo **once** (skip if you already have `cmapss-rul/`):

```bash
git clone https://huggingface.co/spaces/alvinalias/cmapss-rul
```

4. Copy staged deploy files into the HF clone (from **repo root**):

```bash
rsync -av deploy/hf_space/ cmapss-rul/ \
  --exclude sync_to_space.sh \
  --exclude PUSH_TO_HF.md \
  --exclude OPTIONAL_WAKE_STRATEGIES.md
```

5. Commit and push to Hugging Face:

```bash
cd cmapss-rul
git add README.md Dockerfile requirements.txt .gitignore api models
git commit -m "Deploy CMAPSS RUL API on HF Spaces"
git push
```

Space URL:

```text
https://alvinalias-cmapss-rul.hf.space
```

## Verify

```bash
curl -s https://alvinalias-cmapss-rul.hf.space/health
```

Expect `model_loaded: true`.

## After HF is live

1. Add `HF_TOKEN` to GitHub repo secrets (same token as retail/industrial):
   https://github.com/aalias01/cmapss-rul-prediction/settings/secrets/actions
2. Push deploy + workflow + frontend to GitHub `main` (Option B, after `/health` is green).
3. Confirm **Sync CMAPSS HF Space** Action runs green.
4. Test [turbofan.alvinalias.com](https://turbofan.alvinalias.com) after Vercel redeploys.
5. Suspend Render `cmapss-rul-api`.

See `docs_local/HF_SPACES_MIGRATION.md` for the full checklist.

## Optional wake strategies

No scheduled wake ping by default. See [`OPTIONAL_WAKE_STRATEGIES.md`](OPTIONAL_WAKE_STRATEGIES.md).
