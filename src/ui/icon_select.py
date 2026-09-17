"""Accessible single-select icon combobox using Gradio's public HTML API."""
from __future__ import annotations

import gradio as gr


SELECT_CSS = """
overflow:visible !important;
.icon-select { position:relative; min-width:0; font:14px/1.4 Arial,sans-serif; color:#f0e6d2; }
.icon-select label { display:block; font-size:13px; font-weight:600; margin-bottom:8px; }
.icon-select-control { display:flex; align-items:center; gap:8px; min-height:44px; padding:0 10px; border:1px solid #37525a; border-radius:4px; background:#06121c; }
.icon-select-control:focus-within { border-color:#0ac8b9; outline:1px solid #0ac8b9; }
.icon-select input { width:100%; min-width:0; flex:1; padding:10px 0; background:transparent; color:#f0e6d2; border:0; outline:none; font:inherit; }
.icon-select input::placeholder { color:#8ca0aa; }
.icon-select button { border:0; background:transparent; color:#c8aa6e; cursor:pointer; padding:6px; }
.icon-select button:focus-visible { outline:2px solid #0ac8b9; }
.icon-select img { width:28px; height:28px; object-fit:contain; flex-shrink:0; }
.icon-select-list { position:absolute; left:0; right:0; top:100%; z-index:100; max-height:245px; overflow-y:auto; background:#0a1925; border:1px solid #8b7443; border-radius:4px; box-shadow:0 12px 24px #0009; padding:4px; margin-top:4px; }
.icon-select [hidden] { display:none !important; }
.icon-select-option { display:flex; align-items:center; gap:10px; padding:7px 9px; min-height:38px; border-radius:3px; cursor:pointer; }
.icon-select-option:hover, .icon-select-option.active { background:#183d48; }
.icon-select-option[aria-selected="true"] { box-shadow:inset 3px 0 #c8aa6e; }
.icon-select-empty { padding:12px; color:#a3b4bc; }
.icon-select-status { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
.division-select .icon-select-option { justify-content:center; }
.division-select .icon-select-option img { width:36px; height:36px; }
.division-select .icon-select-option.has-icon span { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
"""

SELECT_TEMPLATE = """
<div class="icon-select {{#if image_only}}division-select{{/if}}">
  <label for="{{control_id}}">{{field_label}}</label>
  <div class="icon-select-control">
    <img class="selected-icon" alt="" hidden>
    <input id="{{control_id}}" role="combobox" aria-autocomplete="list"
      aria-expanded="false" aria-controls="{{control_id}}-list" autocomplete="off"
      placeholder="{{placeholder}}" {{#unless searchable}}readonly{{/unless}}>
    {{#if optional}}<button type="button" class="clear-choice" aria-label="Clear {{field_label}}" title="Clear selection">×</button>{{/if}}
    <button type="button" class="open-choices" aria-label="Open {{field_label}} choices" tabindex="-1">▾</button>
  </div>
  <div class="icon-select-list" id="{{control_id}}-list" role="listbox" aria-label="{{field_label}} choices" hidden></div>
  <div class="icon-select-status" role="status" aria-live="polite"></div>
</div>
"""

SELECT_JS = """
const choices = props.choices;
const q = (s) => element.querySelector(s);
let opened = false, filtered = [], active = -1;
const normalize = (s) => String(s).toLocaleLowerCase().replace(/[^a-z0-9]/g, '');
function sync() {
    const selected = choices.find(c => c.value === props.value);
    const input = q('input'), icon = q('.selected-icon');
    input.value = selected ? (props.image_only && selected.icon ? '' : selected.label) : '';
    input.placeholder = selected && props.image_only && selected.icon ? '' : props.placeholder;
    input.setAttribute('aria-label', props.field_label + (selected ? ': ' + selected.label : ''));
    input.title = selected ? selected.label : '';
    icon.hidden = !selected?.icon;
    if (selected?.icon) { icon.src = selected.icon; icon.alt = props.image_only ? selected.label : ''; }
}
function close() {
    opened = false; q('.icon-select-list').hidden = true;
    q('input').setAttribute('aria-expanded', 'false');
    q('input').removeAttribute('aria-activedescendant');
    sync();
}
function highlight() {
    q('.icon-select-list').querySelectorAll('[role=option]').forEach((row, i) => row.classList.toggle('active', i === active));
    const row = q('.icon-select-list').querySelector('.active');
    if (row) {
        q('input').setAttribute('aria-activedescendant', row.id);
        row.scrollIntoView({block:'nearest'});
    } else q('input').removeAttribute('aria-activedescendant');
}
function open(query = '') {
    opened = true;
    filtered = choices.filter(c => normalize(c.label + ' ' + c.value).includes(normalize(query)));
    const list = q('.icon-select-list');
    list.replaceChildren(); list.hidden = false;
    q('input').setAttribute('aria-expanded', 'true');
    filtered.forEach((choice, i) => {
        const row = document.createElement('div');
        row.className = 'icon-select-option' + (choice.icon ? ' has-icon' : '');
        row.id = props.control_id + '-option-' + i; row.dataset.index = i;
        row.setAttribute('role', 'option'); row.setAttribute('aria-label', choice.label);
        row.setAttribute('aria-selected', String(choice.value === props.value)); row.title = choice.label;
        if (choice.icon) { const img = document.createElement('img'); img.src = choice.icon; img.alt = ''; img.loading = 'lazy'; row.append(img); }
        const text = document.createElement('span'); text.textContent = choice.label; row.append(text);
        list.append(row);
    });
    if (!filtered.length) { const empty = document.createElement('div'); empty.className = 'icon-select-empty'; empty.textContent = 'No recorded champions match.'; list.append(empty); }
    active = filtered.findIndex(c => c.value === props.value);
    if (active < 0 && filtered.length) active = 0;
    q('.icon-select-status').textContent = filtered.length + ' options available';
    highlight();
}
function choose(choice) {
    props.value = choice.value;
    close(); trigger('input');
}
element.addEventListener('input', event => {
    if (event.target !== q('input')) return;
    // Typing filters the menu; only a listed option commits a value.
    open(event.target.value);
});
element.addEventListener('click', event => {
    const option = event.target.closest('[role=option]');
    if (option) { choose(filtered[Number(option.dataset.index)]); q('input').focus(); return; }
    if (event.target.closest('.clear-choice')) { choose({value:''}); q('input').focus(); return; }
    if (event.target === q('input') || event.target.closest('.open-choices')) {
        if (opened) close(); else { open(); q('input').focus(); q('input').select(); }
    }
});
element.addEventListener('mousedown', event => {
    if (event.target.closest('[role=option]')) event.preventDefault();
});
element.addEventListener('focusout', event => { if (!element.contains(event.relatedTarget)) close(); });
element.addEventListener('keydown', event => {
    if (event.target !== q('input')) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        if (!opened) { open(); return; }
        active = filtered.length ? (active + (event.key === 'ArrowDown' ? 1 : -1) + filtered.length) % filtered.length : -1;
        highlight();
    } else if (event.key === 'Enter' && opened) {
        event.preventDefault(); if (active >= 0) choose(filtered[active]);
    } else if (event.key === 'Escape') { event.preventDefault(); close(); }
    else if (event.key === 'Tab') close();
});
watch('value', () => { sync(); });
sync();
"""


def icon_select(choices: list[dict], *, label: str, value: str = "", optional: bool = False,
                searchable: bool = False, image_only: bool = False, elem_id: str) -> gr.HTML:
    return gr.HTML(
        value=value, label=label, html_template=SELECT_TEMPLATE, css_template=SELECT_CSS,
        js_on_load=SELECT_JS, choices=choices, field_label=label, control_id=elem_id + "-input",
        optional=optional, searchable=searchable, image_only=image_only,
        placeholder="Search recorded champions…" if searchable else "Choose…",
        elem_id=elem_id, scale=1, min_width=160,
    )
