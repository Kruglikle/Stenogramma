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
})();
