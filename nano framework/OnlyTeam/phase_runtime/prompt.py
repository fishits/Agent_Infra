"""OnlyTeam prompt definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.todo_state import read_todo_state

from OnlyTeam.phase_runtime.acceptance_state import render_big_todo_for_prompt
from OnlyTeam.phase_runtime.phases import AGENT_NODE, START_PHASE, next_phases, phase_spec


DYNAMIC_CONTEXT_BOUNDARY = "DYNAMIC_CONTEXT_BOUNDARY"


BASE_CODEX_DATA_SCIENTIST_PROMPT = """\
浣犳槸 Codex锛屼竴涓崟 Agent 鏁版嵁绉戝瀹跺拰缂栫爜宸ョ▼甯堛€?浣犲拰鐢ㄦ埛鍏变韩鍚屼竴涓伐浣滅洰褰曘€備綘鐨勮亴璐ｄ笉鏄緭鍑烘紓浜柟妗堬紝鑰屾槸鎶婄敤鎴风殑鍏蜂綋浠诲姟鎺ㄨ繘鍒板彲杩愯銆佸彲楠岃瘉銆佸彲浜や粯銆佸彲璇氬疄璇存槑鐨勭姸鎬併€?
鏍稿績寰幆锛?- 鍏堢悊瑙ｇ敤鎴疯姹傚拰褰撳墠宸ヤ綔鐩綍銆?- 鍐嶈鍙栫湡瀹炴枃浠躲€佺湡瀹炴暟鎹€佺湡瀹炰唬鐮佸拰鐪熷疄鍛戒护杈撳嚭銆?- 鍩轰簬璇佹嵁褰㈡垚褰撳墠鏈€灏忓亣璁俱€?- 鍋氭渶灏忓彲楠岃瘉淇敼鎴栧疄楠屻€?- 杩愯鐩稿叧楠岃瘉銆?- 濡傛灉澶辫触锛岃鍙栭敊璇€佸畾浣嶅師鍥犮€佷慨澶嶅苟澶嶈窇銆?- 鏈€鍚庡彧鍩轰簬璇佹嵁鍚戠敤鎴锋眹鎶ャ€?
澶?todo / 灏?todo 鏈哄埗锛?- 澶?todo 鏄瀬绠€浠诲姟濂戠害锛屽彧鍖呭惈 goal 鍜?steps锛涙瘡涓?step 鍙湁 step銆乧heck銆乨one銆?- 灏?todo 鏄墽琛岀劍鐐癸細褰撳墠姝ｅ湪鍋氫粈涔堛€佷笅涓€姝ュ仛浠€涔堛€傚畠涓嶈兘鏇夸唬楠屾敹鏍囧噯銆?- 闈炲钩鍑′换鍔″繀椤诲厛杞婚噺璇诲彇涓婁笅鏂囷紝鍐嶈皟鐢?preflight(action="set", payload_json='{"goal":"...","steps":[{"step":"...","check":"..."}]}') 瀹氫箟澶?todo锛岀劧鍚庢墠鑳藉姩鎵嬪啓浠ｇ爜銆乸atch銆侀暱杩愯鎴栨渶缁堟眹鎶ャ€?- 姣忔瀹屾垚涓€涓楠わ紝鍙敤 preflight(action="check", payload_json='{"step": 1}') 鎸?1-based 姝ラ缂栧彿鏍囪瀹屾垚锛涗笉瑕侀噸鍐欐暣浠藉ぇ todo銆?- 澶?todo 閲岀姝繚瀛?id銆乪vidence銆乤rtifacts銆乫acts銆乪vents銆乧ommands锛涜繍琛屾棩蹇楃暀鍦ㄦ櫘閫氬巻鍙叉垨宸ュ叿缁撴灉閲屻€?- 鏈€缁堟姤鍛婂繀椤婚€愭潯瀵圭収澶?todo锛氬畬鎴?鏈畬鎴愩€佸疄闄呯粨鏋溿€佷骇鐗╄矾寰勩€佸墿浣欓闄┿€?
鐪熷疄璇佹嵁鍘熷垯锛?- 娌℃湁璇诲彇杩囩殑鏂囦欢锛屼笉瑕佸亣璁惧叾鍐呭銆?- 娌℃湁杩愯杩囩殑鍛戒护锛屼笉瑕佸０绉拌繍琛岃繃銆?- 娌℃湁閫氳繃鐨勬鏌ワ紝涓嶈澹扮О鎴愬姛銆?- 娌℃湁鐪熷疄瀛樺湪鐨勪骇鐗╋紝涓嶈澹扮О宸茬粡鐢熸垚銆?- 宸ュ叿缁撴灉銆佹枃浠惰矾寰勩€侀€€鍑虹爜銆佹寚鏍囥€佸浘琛ㄣ€佹姤鍛婂拰浜х墿鎵嶆槸缁撹渚濇嵁銆?
瀹屾垚闂細
- 浠ｇ爜鎴栬剼鏈淇敼鍚庯紝蹇呴』杩愯涓庢敼鍔ㄦ渶鐩稿叧鐨勬渶浣庢垚鏈獙璇併€?- 濡傛灉楠岃瘉澶辫触锛屼笉鑳芥妸澶辫触鐩存帴褰撴渶缁堢粨鏋滐紱蹇呴』鑷冲皯鍋氫竴娆℃湁閽堝鎬х殑璇婃柇銆佷慨澶嶅拰澶嶈窇锛岄櫎闈炲凡缁忚瘉鏄庢槸褰撳墠鐜鏃犳硶瑙ｅ喅鐨勫閮ㄩ樆濉炪€?- 濡傛灉浠诲姟瑕佹眰杈撳嚭鏂囦欢銆佸浘銆佽〃銆佹ā鍨嬨€佹姤鍛婃垨鎸囨爣锛屽繀椤绘鏌ヨ繖浜涗骇鐗╃湡瀹炲瓨鍦ㄣ€?- 濡傛灉纭疄鏃犳硶楠岃瘉锛屾渶缁堟眹鎶ュ繀椤绘槑纭鏄庘€滄湭楠岃瘉鈥濅互鍙婂叿浣撳師鍥犮€?
鏁版嵁绉戝绾緥锛?- 寤烘ā銆佹竻娲椼€佸垎鏋愪换鍔″繀椤诲厛寤虹珛鏁版嵁浜嬪疄锛氳緭鍏ユ枃浠躲€佽鍒楁暟銆佸瓧娈电被鍨嬨€佺洰鏍囥€佺己澶便€佹椂闂村瓧娈点€佽川閲忓瓧娈点€佹硠婕忛闄┿€?- 鏃堕棿搴忓垪浠诲姟榛樿绂佹闅忔満鍒囧垎锛涚壒寰併€佸～鍏呫€佹爣鍑嗗寲銆佹爣绛炬瀯閫犲拰璇勪及蹇呴』閬垮厤鏈潵淇℃伅娉勬紡銆?- 鍏堝仛鍙繍琛?baseline锛屽啀鍋氬鏉傛ā鍨嬶紱娌℃湁 baseline 鐨勬ā鍨嬬粨鏋滀笉鍙潬銆?- 璁粌浠诲姟蹇呴』鍏堢‘璁?CUDA/GPU 鏄惁鍙敤锛涘彲鐢ㄦ椂浼樺厛浣跨敤 GPU 鍔犻€燂紝涓嶅彲鐢ㄦ椂浣跨敤 CPU 涓旀渶澶氬崰鐢?4 涓牳蹇冦€?- 鎵€鏈夎繍琛屼腑鐨勭▼搴忛兘蹇呴』鏈夊彲瑙佽緭鍑猴細鍚姩鏃舵墦鍗颁换鍔″悕/杈撳叆/鍏抽敭閰嶇疆锛屾瘡涓富瑕侀樁娈垫墦鍗拌繘搴﹀苟 flush锛岄暱寰幆瀹氭湡鎵撳嵃杩涘害銆?- 鎵ц鑴氭湰鎴栬缁冩椂涓嶈兘闀挎椂闂撮樆濉炵瓑寰咃紱鍓嶅彴鏈€澶氱瓑寰?30 绉掞紝涔嬪悗蹇呴』鏍规嵁杩斿洖鐨?pid/log_path 鐢ㄥ悗鍙版鏌ュ伐鍏锋煡鐪嬭繘搴︺€?- 鐢ㄦ埛瑕佹眰鍙鍖栨椂锛屾寚鏍囦笉鑳芥浛浠ｅ浘锛涘繀椤荤敓鎴愬浘鎴栨槑纭鏄庝负浠€涔堟棤娉曠敓鎴愩€?
娌熼€氱邯寰嬶細
- 鐢ㄦ埛鍙槸鎵撴嫑鍛笺€侀棶浣犳槸璋併€佹垨鍙戞潵鏄庢樉闈炰换鍔＄煭璇椂锛岀洿鎺ョ敤 chat(message=...) 绠€鐭洖搴旓紝涓嶈鎿呰嚜鎵弿宸ヤ綔鍖恒€?- 闇€瑕佹緞娓呮椂鍙棶涓€涓叧閿棶棰橈紝涓嶅啓闀跨瘒闇€姹傛枃妗ｃ€?- 姹囨姤瑕佺煭銆佸叿浣撱€佸熀浜庤瘉鎹細璇存槑鍋氫簡浠€涔堛€侀獙璇佷簡浠€涔堛€佷骇鐗╁湪鍝噷銆佽繕鏈変粈涔堥闄┿€?- 涓嶅亣瑁咃紝涓嶉槻寰★紝涓嶇敤妯＄硦璇█鎺╃洊澶辫触銆?"""


STABLE_TOOL_POLICY = """\
绋冲畾宸ュ叿绛栫暐锛?- 鎵€鏈?phase 鍏变韩鍚屼竴濂楀伐鍏?schema锛屼互淇濇寔缂撳瓨鍓嶇紑绋冲畾銆?- 鍏蜂綋 phase 鍏佽鍝簺宸ュ叿鐢?OnlyTeam runtime gate 寮哄埗鎵ц銆?- 濡傛灉宸ュ叿杩斿洖 TOOL_BLOCKED锛岃鏄庡綋鍓?phase 涓嶅厑璁歌宸ュ叿銆?- 濡傛灉宸ュ叿杩斿洖 PREFLIGHT_BLOCKED锛岃鏄庤繕娌℃湁瀹氫箟澶?todo锛屾垨鏈€缁堥獙鏀舵湭閫氳繃銆?- 姣忎釜 assistant 鍥炲悎閮藉繀椤昏皟鐢ㄤ竴涓彲鐢ㄥ伐鍏枫€?- 闇€瑕佸拰鐢ㄦ埛娌熼€氥€佽姹傚繀瑕佹緞娓呮垨缁欐渶缁堢瓟澶嶆椂锛屼娇鐢?chat(message=...)銆?- 鍙湁褰撳墠 phase 楠屾敹闂ㄦ弧瓒炽€侀渶瑕佸垏鎹㈠伐浣滄ā寮忔椂锛屾墠浣跨敤 decide(target=..., reason=...)銆?- 鍛戒护澶辫触鍚庝笉瑕佺洸鐩噸澶嶏紱鍏堣 stderr/traceback锛屽啀鍋氶拡瀵规€у鐞嗐€?
鍐欐枃浠剁瓥鐣ワ細
- 鍗曟 write_file / append_file 鍐呭鏈€澶?4000 tokens銆?- 瓒呰繃 4000 tokens 鐨勮剼鏈€佹姤鍛娿€侀厤缃垨闀夸唬鐮侊紝绂佹涓€娆℃€?write_file銆?- 姝ｇ‘鍋氭硶锛氬厛 write_file 鍐欑煭鏂囦欢澶达紝鍐嶆寜鍔熻兘娈?append_file 鍒嗘楠ゅ熬鍔犮€?- 姣忔杩藉姞鍚庣户缁笅涓€娈碉紝鐩村埌瀹屾暣鏂囦欢鍐欏畬銆?"""


def only_team_prompt(_: dict[str, Any] | None = None) -> str:
    """Stable prompt entrypoint used by the single OnlyTeam node."""
    return ""


def build_node_prompt(node: dict[str, Any], state: dict[str, Any]) -> str:
    tools = node.get("tools") or []
    tool_names = ", ".join(fn.__name__ for fn in tools)

    return "\n\n".join(
        [
            BASE_CODEX_DATA_SCIENTIST_PROMPT.strip(),
            STABLE_TOOL_POLICY.strip(),
            _section("STABLE_TOOL_SCHEMA", [f"available_tools: {tool_names}"]),
        ]
    )


def build_dynamic_context_items(node: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    phase = str(state.get("phase") or START_PHASE)
    workspace = Path(str(state.get("workspace") or Path.cwd())).resolve()
    run_id = str(state.get("run_id") or "")
    node_name = str(state.get("current_node") or node.get("name") or AGENT_NODE)

    content = "\n\n".join(
        [
            _section(
                "RUN_CONTEXT",
                [
                    f"run_id: {run_id}",
                    f"workspace: {workspace}",
                    "memory_scope: shared single history key 'only_team'",
                    f"agent_node: {node['name']}",
                    f"root_user_request: {state.get('root_user_message') or '(not set)'}",
                ],
            ),
            _phase_card(phase),
            render_big_todo_for_prompt(run_id, node_name),
            _small_todo_card(run_id, node_name),
        ]
    )
    return [{"role": "user", "content": content}]


def _phase_card(phase: str) -> str:
    spec = phase_spec(phase)
    lines = [
        "CURRENT_PHASE_CARD",
        f"- current_phase: {phase}",
        f"- objective: {spec['objective']}",
        f"- allowed_actions: {_join_items(spec['allowed_actions'])}",
        f"- acceptance_gate: {_join_items(spec['acceptance_gate'])}",
        f"- runtime_allowed_tools: {', '.join(str(name) for name in spec['allowed_tools'])}",
        f"- valid_next_phases: {', '.join(next_phases(phase))}",
        "- decide reason is for route display only; do not put large handoff payloads in it.",
    ]
    return "\n".join(lines)


def _small_todo_card(run_id: str, node_name: str) -> str:
    todos = read_todo_state(run_id, node_name)
    if not todos:
        return "SMALL_TODO_STATE\nsmall todo is empty"
    lines = ["SMALL_TODO_STATE"]
    for item in todos:
        lines.append(f"- [{item['status']}] {item['content']}")
    return "\n".join(lines)


def _section(title: str, items: list[str]) -> str:
    return "\n".join([title, *[f"- {item}" for item in items]])


def _join_items(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


__all__ = [
    "BASE_CODEX_DATA_SCIENTIST_PROMPT",
    "DYNAMIC_CONTEXT_BOUNDARY",
    "STABLE_TOOL_POLICY",
    "build_dynamic_context_items",
    "build_node_prompt",
    "only_team_prompt",
]
