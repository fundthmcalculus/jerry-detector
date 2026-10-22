const list = document.getElementById("clips");
const status = document.getElementById("status");

function card(clip) {
  const li = document.createElement("li");
  li.className = "card";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = `Crash on ${clip.camera}`;
  img.src = `/clips/${encodeURIComponent(clip.id)}.gif`;

  const caption = document.createElement("div");
  caption.className = "caption";
  const when = new Date(clip.created).toLocaleString();
  caption.textContent = `${clip.camera} · ${when}`;

  li.append(img, caption);
  return li;
}

fetch("/api/clips")
  .then((r) => {
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  })
  .then((clips) => {
    if (clips.length === 0) {
      status.textContent = "No clips yet.";
      return;
    }
    status.remove();
    list.replaceChildren(...clips.map(card));
  })
  .catch(() => {
    status.textContent = "Could not load clips.";
  });
