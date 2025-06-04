// content.js

// 1) Строгая регулярка: ловит ровно "домен(.суффикс)(/путь)?" + необязательный http(s) и ничего кроме него.
//    Примеры попаданий: "kazindoor.kz", "sub.example.com/path", "https://google.com".
//    Примеры непропаданий: "hello world", "kazindoor", " kazindoor.kz extra " и т.п.
const urlRegex = /^((https?:\/\/)?([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}(\/\S*)?)$/i;

// 2) Настройка вашего локального API
const API_ENDPOINT = 'http://127.0.0.1:5000/check_url';
async function checkLink(url) {
    try {
        const res = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();
        return data.status || 'unknown';
    } catch (err) {
        console.error('Ошибка при checkLink:', err);
        return 'unknown';
    }
}

// 3) Функция-помощник, которая для данного текстового узла (Node.TEXT_NODE) возвращает его родительский элемент,
//    если этот текстовый узел «листьевой» (то есть он целиком совпадает с urlRegex и не содержит пробелов),
//    а также этот родитель ещё не обработан (нет data-link-processed).
function processTextNodeIfUrl(textNode) {
    const txtRaw = textNode.nodeValue;
    if (!txtRaw) return null;
    const txt = txtRaw.trim();
    const m = txt.match(urlRegex);
    if (!m || m[0] !== txt) {
        // Не целиком URL → ничего не делаем
        return null;
    }

    const parent = textNode.parentElement;
    if (!parent) return null;

  // 3.1) Проверяем, что внутри parent нет уже наших бейджей:
    if (parent.dataset.linkProcessed === 'true') {
        return null;
    }

  // 3.2) Проверяем: родительский элемент НЕ должен иметь других узлов кроме этого textNode + (в будущем) бейджа.
  // Например, если внутри родителя есть дополнительный <span>foo</span> после текста,
  // то parent.textContent даст что-то вроде "kazindoor.kzfoo", не совпадающее с urlRegex.
  // Но TreeWalker сработал на leaf textNode = "kazindoor.kz", и это безопасно.
  //
  // Другими словами, мы убеждаемся, что textNode — единственный «нескрытый» текст в этом parent.
    const siblings = Array.from(parent.childNodes).filter(node => {
        // Если это текстовый узел, проверим, есть ли в нем символы кроме пробелов
        if (node.nodeType === Node.TEXT_NODE) {
            return node.nodeValue.trim() !== '';
        }
        // Если это элемент (например, <span class="link-badge">), его мы тоже учитываем,
        // но мы его ещё не вставляли (его не будет при первой проверке).
        return node.nodeType === Node.ELEMENT_NODE && !node.classList.contains('link-badge');
    });

  // Если помимо нашего TEXT_NODE в parent есть хоть ещё один «не бейджевой» дочерний узел — откажемся,
  // потому что это значит, что parent содержит не только «kazindoor.kz», а, например, «kazindoor.kz extra»,
  // либо «<span>kazindoor.kz</span><span>что-то ещё</span>», поэтому не целиком URL.
    if (siblings.length > 1) {
        return null;
    }

  // Всё ок — возвращаем parent, чтобы туда вставить бейдж
    return parent;
}

// 4) Основная функция: обходим все текстовые узлы (TreeWalker) и вставляем бейджи в те, что должны.
async function addLinkBadges() {
  // 4.1) Создаём TreeWalker, который проходит по всем текстовым узлам на странице:
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    let node;
    while ((node = walker.nextNode())) {
        // Для каждого текстового узла пытаемся получить «родительский URL-контейнер»
        const parentContainer = processTextNodeIfUrl(node);
        if (!parentContainer) continue;

        // 4.2) Теперь мы уверены: parentContainer — именно тот <span> или <div> или <a>,
        //      внутри которого единственный текст — «kazindoor.kz» (или любой другой домен).
        //      Вставляем бейдж внутрь parentContainer:
        parentContainer.dataset.linkProcessed = 'true';
        const badge = document.createElement('span');
        badge.classList.add('link-badge');
        badge.style.backgroundColor = 'gray'; // по умолчанию серый

        // appendChild поместит бейдж после текстового узла внутри родителя
        parentContainer.appendChild(badge);

        // 4.3) Асинхронно запрашиваем статус у сервера и меняем цвет бейджа:
        (async () => {
            const status = await checkLink(node.nodeValue.trim());
            badge.dataset.linkStatus = status;

            switch (status) {
                case 'benign':
                    badge.style.backgroundColor = 'green';
                    break;
                case 'phishing':
                case 'malware':
                    badge.style.backgroundColor = 'red';
                    break;
                case 'defacement':
                case 'suspicious':
                    badge.style.backgroundColor = 'orange';
                    break;
                default:
                    badge.style.backgroundColor = 'gray';
            }
        })();
    }
}

// 5) После того как страница «грузится», сразу запускаем addLinkBadges() и вешаем MutationObserver,
//    чтобы при появлении новых комментариев (или при переработке DOM) снова вызывать ту же функцию:
window.addEventListener('load', () => {
    addLinkBadges();

    const observer = new MutationObserver(() => {
        // При любых изменениях в subtree тела документа вызываем addLinkBadges,
        // чтобы захватить вновь появившиеся текстовые узлы-домены.
        addLinkBadges();
    });
    observer.observe(document.body, { childList: true, subtree: true });
});