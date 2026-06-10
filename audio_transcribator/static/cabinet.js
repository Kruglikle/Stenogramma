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

  const setUploadProgress = (form, percent, label) => {
    const progress = form.querySelector("[data-upload-progress]");
    const bar = form.querySelector("[data-upload-progress-bar]");
    const percentText = form.querySelector("[data-upload-progress-percent]");
    const labelText = form.querySelector("[data-upload-progress-label]");
    if (!progress || !bar || !percentText || !labelText) return;

    progress.hidden = false;
    progress.classList.toggle("is-indeterminate", percent === null);
    if (percent === null) {
      percentText.textContent = "";
      bar.style.width = "";
    } else {
      const normalized = Math.max(0, Math.min(100, Math.round(percent)));
      percentText.textContent = `${normalized}%`;
      bar.style.width = `${normalized}%`;
    }
    labelText.textContent = label;
  };

  const setUploadSubmitting = (form, isSubmitting) => {
    const submitButton = form.querySelector("[data-upload-submit]");
    for (const element of form.elements) {
      if (element === submitButton) continue;
      element.disabled = isSubmitting;
    }
    if (submitButton) {
      submitButton.disabled = isSubmitting;
      submitButton.textContent = isSubmitting ? "Загружаю..." : "Запустить обработку";
    }
  };

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-upload-form]");
    if (!form) return;

    event.preventDefault();
    if (form.dataset.submitting === "true") return;
    form.dataset.submitting = "true";

    // FormData must be created before disabling fields: disabled fields are not submitted.
    const formData = new FormData(form);
    const hasFile = Array.from(form.querySelectorAll('input[type="file"]')).some((input) => input.files.length > 0);
    setUploadSubmitting(form, true);
    setUploadProgress(form, hasFile ? 0 : null, hasFile ? "Загрузка файла..." : "Отправляю задачу...");

    const xhr = new XMLHttpRequest();
    xhr.open(form.method || "POST", form.action);
    xhr.upload.addEventListener("progress", (progressEvent) => {
      if (!progressEvent.lengthComputable) {
        setUploadProgress(form, null, "Загрузка файла...");
        return;
      }
      setUploadProgress(form, (progressEvent.loaded / progressEvent.total) * 100, "Загрузка файла...");
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 400) {
        setUploadProgress(form, 100, "Загрузка завершена. Открываю задачу...");
        window.location.href = xhr.responseURL || "/ui/upload";
        return;
      }

      form.dataset.submitting = "false";
      setUploadSubmitting(form, false);
      document.open();
      document.write(xhr.responseText);
      document.close();
    });
    xhr.addEventListener("error", () => {
      form.dataset.submitting = "false";
      setUploadSubmitting(form, false);
      setUploadProgress(form, null, "Не удалось отправить задачу. Проверьте соединение.");
    });
    xhr.send(formData);
  });
})();
