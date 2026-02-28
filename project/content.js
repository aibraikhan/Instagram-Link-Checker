// content.js

// 0) Паттерн для поиска URL-подстрок в любом тексте
const inlineUrlPattern = /((?:https?:\/\/)?(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6}(?:\/\S*)?)/g;

// 1) Общий тултип
const tooltip = document.createElement('div');
tooltip.id = 'link-tooltip';
// кнопка обновления добавится динамически
document.body.appendChild(tooltip);

function normalizeUrl(url) {
    // Проверяем наличие схемы https:// или http://, если нет, добавляем https://
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
    }
    return url;
}


// 2) Нормализация и проверка “тут ли URL” (для чистых узлов, оставлено как есть)
function parseIfUrl(txt) {
    const trimmed = txt.trim();
    if (!trimmed || !trimmed.includes('.')) return null;
    try {
        const url = /^https?:\/\//i.test(trimmed)
            ? new URL(trimmed)
            : new URL('https://' + trimmed);
        const norm = url.href.replace(/^https?:\/\//i, '').replace(/\/$/, '');
        const raw  = trimmed.replace(/^https?:\/\//i, '').replace(/\/$/, '');
        return norm === raw ? url : null;
    } catch {
        return null;
    }
}

// 3) Отправка на проверку в background
function checkLink(url) {
    return new Promise(resolve => {
        chrome.runtime.sendMessage(
            { action: 'checkLink', url },
            response => resolve(response?.status || 'unknown')
        );
    });
}

// 4) Обёртываем inline-URL-подстроки в span[data-link-processed]
function markInlineLinks(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
        // пропускаем уже обработанные
        if (node.parentElement.matches('span[data-link-processed]')) continue;
        if (inlineUrlPattern.test(node.nodeValue)) {
            textNodes.push(node);
        }
    }

    for (const textNode of textNodes) {
        const parent = textNode.parentNode;
        const text   = textNode.nodeValue;
        inlineUrlPattern.lastIndex = 0;
        const frag = document.createDocumentFragment();
        let lastIdx = 0, m;
        while (m = inlineUrlPattern.exec(text)) {
            // текст до URL
            if (m.index > lastIdx) {
                frag.appendChild(document.createTextNode(text.slice(lastIdx, m.index)));
            }

            // проверяем, а действительно ли это URL?
            const candidate = m[0];
            const parsed    = parseIfUrl(candidate);
            if (parsed) {
                // да — настоящий URL
                const urlSpan = document.createElement('span');
                urlSpan.textContent           = candidate;
                urlSpan.dataset.linkProcessed = 'true';
                urlSpan.dataset.linkUrl       = parsed.href.replace(/\/$/, '');
                frag.appendChild(urlSpan);
            } else {
                // нет — просто вставляем текст как есть
                frag.appendChild(document.createTextNode(candidate));
            }

            lastIdx = inlineUrlPattern.lastIndex;
        }
        // остаток после последнего URL
        if (lastIdx < text.length) {
            frag.appendChild(document.createTextNode(text.slice(lastIdx)));
        }
        parent.replaceChild(frag, textNode);
    }
}

// 5) Помощник для «чистых» узлов-URL
function processTextNodeIfUrl(textNode) {
    const parent = textNode.parentElement;
    if (!parent || parent.dataset.linkProcessed) return null;

    // Проверяем, что весь текст в узле — потенциальный URL
    const urlObj = parseIfUrl(textNode.nodeValue);
    if (!urlObj) return null;

    // Убеждаемся, что в parent нет другого «живого» текста
    const others = Array.from(parent.childNodes).filter(n =>
        n !== textNode &&
        n.nodeType === Node.TEXT_NODE &&
        n.nodeValue.trim()
    );
    if (others.length) return null;

    // Всё ок — возвращаем parent и домен для дальнейшей проверки
    parent.dataset.linkProcessed = 'true';
    parent.dataset.linkUrl = urlObj.href.replace(/\/$/, '');
    return parent;
}

// 6) Вставляем бейджи для всех span[data-link-processed]
async function addLinkBadges() {
    try{
        // сначала обрабатываем inline ссылки
        markInlineLinks(document.body);

        // а теперь проходим по уже размеченным span
        const spans = document.querySelectorAll('span[data-link-processed]');
        for (const span of spans) {
            if (span.dataset.linkChecked) continue;
            span.dataset.linkChecked = 'true';

            const badge = document.createElement('span');
            badge.classList.add('link-badge');
            span.appendChild(badge);

            // ключевая поправка: берем домен из dataset того же контейнера
            const urlToCheck = span.dataset.linkUrl;
            console.log('▶️ addLinkBadges(), checking URL:', urlToCheck);

            const resp = await checkLink(urlToCheck);
            console.log('⬅️ checkLink response for', urlToCheck, '→', resp);
            console.log('checkLink returned:', resp);
            // если resp — объект, достаём поле status
            const status = (resp && typeof resp === 'object' && 'status' in resp)
                ? resp.status
                : resp;
            badge.dataset.linkStatus = status;

            badge.addEventListener('mouseenter', () => {
                const status = badge.dataset.linkStatus || 'unknown';
                tooltip.textContent      = status;
                tooltip.style.background = ({
                    benign:    'green',
                    phishing:  'red',
                    defacement:'orange',
                    malware:   'red',
                    malicious: 'red',
                    unknown:   'grey'
                })[status] || 'grey';
                tooltip.style.display = 'block';

                const r    = badge.getBoundingClientRect();
                let top    = r.top - tooltip.offsetHeight - 6;
                if (top < 4) top = r.bottom + 6;
                let left   = r.left + (r.width - tooltip.offsetWidth)/2;
                if (left < 4) left = 4;
                if (left + tooltip.offsetWidth > window.innerWidth - 4) {
                    left = window.innerWidth - tooltip.offsetWidth - 4;
                }
                tooltip.style.left = `${left}px`;
                tooltip.style.top  = `${top}px`;
            });
            badge.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });
        }
    } catch (err) {
        console.error('addLinkBadges failed:', err);
    }
}

// 7) Debounce + наблюдение за DOM
let badgeTimer;
function scheduleBadges() {
    clearTimeout(badgeTimer);
    badgeTimer = setTimeout(addLinkBadges, 150);
}

window.addEventListener('load', scheduleBadges);
new MutationObserver(scheduleBadges)
    .observe(document.body, { childList: true, subtree: true });