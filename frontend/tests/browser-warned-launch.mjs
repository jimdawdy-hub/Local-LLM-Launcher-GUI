// Built UI acceptance: a memory-only "won't fit" verdict leaves Launch usable as a
// glowing red warning, while hard blockers still disable it. No engine starts.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `warned-launch-${process.pid}`
function browser(...args) {
  const result = JSON.parse(execFileSync('agent-browser', ['--session', session, '--json', ...args], {encoding:'utf8', timeout:45000}))
  assert.equal(result.success, true, result.error)
  return result.data
}
try {
  browser('open', url)
  console.log(browser('eval', `(async () => {
    const wait = ms => new Promise(r => setTimeout(r, ms));
    const check = (v, message) => { if (!v) throw new Error(message); };
    const original = window.fetch.bind(window);
    let verdict = {level:'red', headline:'Won\\'t fit — needs ~40 GB but only ~29 GB is available.', details:[], override:true};
    let launches = 0;
    const model = {repo_id:'test/big', name:'Test Big', path:'/fake/big.gguf', format:'gguf', size_bytes:1073741824, gguf_files:[]};
    window.fetch = async (path, options) => {
      if (path === '/api/hardware') {
        const hw = await (await original(path, options)).json();
        return Response.json({...hw, engines:{...hw.engines, llamacpp_path:'/fake/llama-server'}});
      }
      if (path === '/api/models') return Response.json({models:[model]});
      if (path.startsWith('/api/presets')) return Response.json({presets:[{name:'Test',config:{}}]});
      if (path === '/api/advise') return Response.json({overall:verdict, budget:{needed_gb:40, available_gb:29}, flags:{}});
      if (path === '/api/servers' && options?.method === 'POST') {
        launches++;
        return Response.json({detail:'Synthetic check: no engine started.'}, {status:400});
      }
      return original(path, options);
    };
    const refresh = [...document.querySelectorAll('button')].find(b=>b.textContent==='Refresh');
    for (let attempt = 0; attempt < 2; attempt++) {
      refresh.click();
      await wait(50);
      const deadline = Date.now() + 10000;
      while (refresh.disabled && Date.now() < deadline) await wait(25);
      check(!refresh.disabled, 'Hardware refresh timed out');
    }
    [...document.querySelectorAll('nav button')].find(b=>b.textContent.endsWith('Launch')).click();
    await wait(900);
    const btn = () => document.querySelector('.launchbtn');
    check(!btn().disabled, 'Memory-only red verdict still blocks launch');
    check(btn().classList.contains('warned'), 'Warned button lacks its red style');
    check(btn().textContent.trim() === 'BAD IDEA – YOU HAVE BEEN WARNED', 'Wrong warned label: ' + btn().textContent);
    check(getComputedStyle(btn()).animationName !== 'none' || matchMedia('(prefers-reduced-motion: reduce)').matches, 'Warned button does not glow');
    btn().click();
    await wait(300);
    check(launches === 1, 'Warned button did not try to launch');
    verdict = {level:'red', headline:'This is a GGUF file; use llama.cpp.', details:[]};
    const advanced = document.querySelector('details');
    const ctx = document.querySelector('input[type="number"]');
    check(ctx, 'No setting to change');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(ctx, String(Number(ctx.value || 4096) + 1));
    ctx.dispatchEvent(new Event('input',{bubbles:true}));
    await wait(900);
    check(btn().disabled, 'Hard blocker permits launch');
    check(!btn().classList.contains('warned'), 'Hard blocker shows the warned button');
    return 'Memory-only red launches as a warning; hard blockers stay disabled';
  })()`).result)
} finally { browser('close') }
