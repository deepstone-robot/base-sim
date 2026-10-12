import DefaultTheme from 'vitepress/theme'
import { useData, useRoute } from 'vitepress'
import { watch, nextTick, onMounted } from 'vue'
import './mermaid.css'

// 文档里 ```mermaid 围栏的客户端渲染。
//
// VitePress 的 shiki 不认识 mermaid，会把它当纯文本高亮，产出一个
// <div class="language-mermaid"><pre class="shiki">…</pre></div>。
// 这里在浏览器里把这种块换成一个真正的 mermaid 图。
//
// 三个已核实的行为细节（都是踩过才会知道的）：
//   1. mermaid 读的是 innerHTML 并对其做 entity 解码 —— 所以赋 textContent 是对的；
//   2. mermaid.run() 会跳过带 data-processed 的元素 —— 重绘前必须清掉该属性；
//   3. mermaid 必须动态 import 且只在 onMounted/watch 里执行 —— SSR 阶段不能碰 document。
export default {
  extends: DefaultTheme,
  setup() {
    const { isDark } = useData()
    const route = useRoute()

    const render = async () => {
      const blocks = Array.from(
        document.querySelectorAll<HTMLElement>('.language-mermaid')
      )
      if (!blocks.length) return

      // 先把源码存下来，再改 DOM —— 明暗切换重绘时要用
      for (const el of blocks) {
        const src = el.dataset.mermaid ?? el.querySelector('code')?.textContent ?? ''
        el.dataset.mermaid = src
        el.textContent = src
        el.classList.add('mermaid')
        el.classList.remove('vp-adaptive-theme') // 去掉代码块外观
        el.removeAttribute('data-processed') // 允许重绘
      }

      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({
          startOnLoad: false,
          theme: isDark.value ? 'dark' : 'default',
        })
        await mermaid.run({ nodes: blocks, suppressErrors: true })
      } catch (err) {
        // 渲染挂了只丢图，退回显示源码，不影响阅读
        console.warn('[mermaid] 渲染失败，已退回显示源码', err)
        for (const el of blocks) {
          el.classList.remove('mermaid')
          el.textContent = el.dataset.mermaid ?? ''
        }
      }
    }

    const schedule = () => nextTick(render)

    onMounted(schedule) // SSR 阶段不会执行
    watch(() => route.path, schedule) // SPA 切换页面
    watch(isDark, schedule) // 明暗切换重绘
  },
}
