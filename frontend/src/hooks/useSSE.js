import { useState, useRef, useCallback } from 'react'

export function useSSE() {
    const [streaming, setStreaming] = useState(false)
    const abortRef = useRef(null)

    const query = useCallback(async ({ question, categoryId, topK = 5, onSources, onToken, onDone }) => {
        setStreaming(true)
        const controller = new AbortController()
        abortRef.current = controller

        try {
            const resp = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, category_id: categoryId ?? null, top_k: topK }),
                signal: controller.signal,
            })

            if (!resp.ok) throw new Error(await resp.text())

            const reader = resp.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop()

                let eventType = ''
                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        eventType = line.slice(6).trim()
                    } else if (line.startsWith('data:')) {
                        const data = JSON.parse(line.slice(5).trim())
                        if (eventType === 'sources') onSources?.(data)
                        else if (eventType === 'token') onToken?.(data)
                        else if (eventType === 'done') onDone?.()
                    }
                }
            }
        } catch (e) {
            if (e.name !== 'AbortError') console.error(e)
        } finally {
            setStreaming(false)
        }
    }, [])

    const abort = useCallback(() => abortRef.current?.abort(), [])

    return { query, streaming, abort }
}