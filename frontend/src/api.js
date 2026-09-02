const BASE = '/api'

export async function getCategories() {
    const r = await fetch(`${BASE}/categories`)
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function createCategory(data) {
    const r = await fetch(`${BASE}/categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function deleteCategory(id) {
    const r = await fetch(`${BASE}/categories/${id}`, { method: 'DELETE' })
    if (!r.ok) throw new Error(await r.text())
}

export async function getDocuments(categoryId) {
    const r = await fetch(`${BASE}/documents?category_id=${categoryId}`)
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function uploadDocument(file, categoryId) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('category_id', categoryId)
    const r = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: fd })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function deleteDocument(id) {
    const r = await fetch(`${BASE}/documents/${id}`, { method: 'DELETE' })
    if (!r.ok) throw new Error(await r.text())
}

export async function getDocumentStatus(id) {
    const r = await fetch(`${BASE}/documents/${id}/status`)
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}