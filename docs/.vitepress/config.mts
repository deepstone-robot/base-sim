import { defineConfig } from 'vitepress'
import { chineseSearchOptimize, pagefindPlugin } from 'vitepress-plugin-pagefind'

// GitHub Pages **项目站**：base 必须以 '/' 开头和结尾。
// nav / sidebar / 正文里的链接一律**不写** base —— VitePress 会自动加前缀；
// 手写会变成 /base-sim/base-sim/...
const base = '/base-sim/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: 'base-sim · MicroDuck',
  description:
    'MicroDuck 小型双足机器人在 mjlab(MuJoCo-Warp + RSL-RL)上的站立 / 坐站 / 行走强化学习任务',

  head: [
    // 注意：head 里的 href **不会**被自动加 base，必须手写。
    // （themeConfig.logo 会自动加，两者行为不同。）
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
  ],

  // 明暗主题切换（VitePress 默认值；想强制暗色就改成 'force-dark'）
  appearance: 'auto',
  lastUpdated: true,

  themeConfig: {
    logo: '/favicon.svg',
    siteTitle: 'base-sim · MicroDuck',

    nav: [
      { text: '首页', link: '/' },
      { text: '架构', link: '/architecture' },
      {
        text: '任务',
        items: [
          { text: '总览', link: '/tasks/' },
          { text: '站立(Stand)', link: '/tasks/stand' },
          { text: '坐站(Sit2Stand)', link: '/tasks/sit2stand' },
          { text: '行走(Walk)', link: '/tasks/walk' },
        ],
      },
      { text: '机器人资产', link: '/robot' },
      {
        text: '奖励设计',
        items: [
          { text: '设计原则', link: '/reward-design' },
          { text: '失败模式史', link: '/failure-modes' },
        ],
      },
      { text: '训练', link: '/training' },
      {
        text: '更多',
        items: [
          { text: '评估与探针', link: '/evaluation' },
          { text: 'sim2real 与域随机化', link: '/sim2real' },
          { text: '测试', link: '/testing' },
          { text: '路线图', link: '/roadmap' },
          { text: '文档工具说明', link: '/tooling' },
        ],
      },
    ],

    // 数组形式的全局侧边栏 = 原来 MkDocs 那种"导航整体留在左侧一列"的布局
    sidebar: [
      { text: '首页', link: '/' },
      { text: '架构', link: '/architecture' },
      {
        text: '任务',
        link: '/tasks/', // 分组标题本身可点（原来的 navigation.indexes）
        collapsed: false,
        items: [
          { text: '总览', link: '/tasks/' },
          { text: '站立(Stand)', link: '/tasks/stand' },
          { text: '坐站(Sit2Stand)', link: '/tasks/sit2stand' },
          { text: '行走(Walk)', link: '/tasks/walk' },
        ],
      },
      { text: '机器人资产', link: '/robot' },
      {
        text: '奖励设计',
        collapsed: false,
        items: [
          { text: '设计原则', link: '/reward-design' },
          { text: '失败模式史', link: '/failure-modes' },
        ],
      },
      { text: '训练', link: '/training' },
      { text: '评估与探针', link: '/evaluation' },
      { text: 'sim2real 与域随机化', link: '/sim2real' },
      { text: '测试', link: '/testing' },
      { text: '路线图', link: '/roadmap' },
      { text: '文档工具说明', link: '/tooling' },
    ],

    // 对应原来的 toc_depth: 3
    outline: { level: [2, 3], label: '本页目录' },

    socialLinks: [{ icon: 'github', link: 'https://github.com/deepstone-robot/base-sim' }],

    editLink: {
      // :path 相对 srcDir(= docs/)，等价于原来的 content.action.edit
      pattern: 'https://github.com/deepstone-robot/base-sim/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },
    lastUpdated: {
      text: '最后更新',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' },
    },
    docFooter: { prev: '上一篇', next: '下一篇' },
    footer: {
      message: '内部工程文档 · 所有数字都来自本仓库的实测脚本',
      copyright: 'Copyright © 2025 deepstone-robot',
    },

    // 主题 UI 的中文化
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色',
    darkModeSwitchTitle: '切换到深色',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
  },

  vite: {
    plugins: [
      // VitePress 自带的 MiniSearch 不认中文分词，所以用 pagefind + 中文分词优化。
      // 插件内部会用 withBase() 加载引擎并给结果路由补 base，无需手工处理；
      // 但**不要**设 manual: true，那样这两件事都得自己做。
      pagefindPlugin({
        forceLanguage: 'zh-cn',
        customSearchQuery: chineseSearchOptimize,
        // 插件默认文案是英文的，这里全部换掉
        btnPlaceholder: '搜索',
        placeholder: '搜索文档',
        emptyText: '没有找到结果',
        loadingText: '搜索中…',
        heading: '共 {{searchResult}} 条结果',
        toSelect: '选择',
        toNavigate: '切换',
        toClose: '关闭',
        searchBy: '搜索提供方',
        displayDetailedList: '显示详细列表',
        resetSearch: '清除搜索',
        closeSearch: '关闭搜索',
      }),
    ],
  },
})
