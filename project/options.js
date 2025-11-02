// options.js

const textarea = document.getElementById('whitelist');
const btn      = document.getElementById('save');
const status   = document.getElementById('status');

function loadOptions() {
    chrome.storage.sync.get({
        localWhitelist: ''
    }, prefs => {
        textarea.value = prefs.localWhitelist;
    });
}

function saveOptions() {
    const list = textarea.value
        .split('\n')
        .map(s => s.trim().toLowerCase())
        .filter(Boolean)
        .join('\n');
    chrome.storage.sync.set({ localWhitelist: list }, () => {
        status.textContent = 'Сохранено!';
        setTimeout(() => status.textContent = '', 1500);
    });
}

btn.addEventListener('click', saveOptions);
document.addEventListener('DOMContentLoaded', loadOptions);
