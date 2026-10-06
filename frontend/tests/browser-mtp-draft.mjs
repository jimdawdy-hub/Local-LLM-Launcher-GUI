// Built UI acceptance: "MTP words drafted ahead" appears only with MTP on and is sent
// with the launch config. Synthetic hardware/model; no engine starts.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `mtp-draft-${process.pid}`
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
    let submitted, launched;
    const model = {repo_id:'test/mtp', name:'Test MTP', path:'/fake/mtp.gguf', format:'gguf', size_bytes:1073741824, gguf_files:[]};
    window.fetch = async (path, options) => {
      if (path === '/api/hardware') {
        const hw = await (await original(path, options)).json();
        return Response.json({...hw, engines:{...hw.engines, llamacpp_path:'/fake/llama-server'}});
      }
      if (path === '/api/models') return Response.json({models:[model]});
      if (path.startsWith('/api/presets')) return Response.json({presets:[{name:'Test',config:{}}]});
      if (path === '/api/advise') {
        submitted = JSON.parse(options.body);
        return Response.json({overall:{level:'green', headline:'Fits.', details:[]}, budget:{needed_gb:10, available_gb:29}, flags:{}});
      }
      if (path === '/api/servers' && options?.method === 'POST') {
        launched = JSON.parse(options.body);
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
    const draft = () => document.querySelector('[aria-label="MTP words drafted ahead"]');
    const mtpToggle = document.querySelector('[aria-label="Use MTP (multi-token prediction)"]');
    check(mtpToggle, 'MTP switch missing');
    check(!draft(), 'Words-ahead box shows while MTP is off');
    mtpToggle.click();
    await wait(600);
    check(draft(), 'Words-ahead box missing with MTP on');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(draft(), '2');
    draft().dispatchEvent(new Event('input',{bubbles:true}));
    await wait(700);
    check(submitted.config.use_mtp === true && submitted.config.mtp_draft_max === 2, 'Advice lacks the words-ahead count: ' + JSON.stringify(submitted.config));
    document.querySelector('.launchbtn').click();
    await wait(400);
    check(launched && launched.config.mtp_draft_max === 2, 'Launch lacks the words-ahead count');
    return 'Words-ahead box follows the MTP switch and reaches advice and launch';
  })()`).result)
} finally { browser('close') }
