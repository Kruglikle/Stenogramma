(() => {
  const openHistoryItem = (event) => {
    const item = event.target.closest("[data-history-url]");
    if (!item || event.target.closest("a")) return;
    window.location.href = item.dataset.historyUrl;
  };

  document.addEventListener("dblclick", openHistoryItem);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    openHistoryItem(event);
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-delete-form]");
    if (!form) return;

    const title = form.dataset.deleteTitle || "этот файл";
    if (!window.confirm(`Удалить "${title}"? Это освободит место, но восстановить результат будет нельзя.`)) {
      event.preventDefault();
    }
  });
})();
