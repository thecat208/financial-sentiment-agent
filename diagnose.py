"""
诊断脚本：逐层定位LLM情感分析的问题环节。

使用方法：
1. 把这个文件放在项目根目录下（和 requirements.txt 同一层）
2. 终端 cd 到项目根目录
3. 运行：python diagnose.py

脚本依次检查：配置加载 → Prompt模板填充 → LM原始调用 → analyze_text() 全链路，
根据每步输出即可定位问题出在哪一层。
"""

print("=" * 60)
print("第1步：检查配置是否正确加载")
print("=" * 60)

try:
    from config.settings import settings
    print(f"当前使用的模型provider: {settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER == "anthropic":
        print(f"当前使用的模型名称: {settings.ANTHROPIC_MODEL}")
        print(f"API Key是否已配置: {'是（长度' + str(len(settings.ANTHROPIC_API_KEY)) + '）' if settings.ANTHROPIC_API_KEY else '否，是空的！'}")
    else:
        print(f"当前使用的模型名称: {settings.OPENAI_MODEL}")
        print(f"API Key是否已配置: {'是（长度' + str(len(settings.OPENAI_API_KEY)) + '）' if settings.OPENAI_API_KEY else '否，是空的！'}")
except Exception as e:
    print(f"❌ 第1步就失败了，配置文件加载有问题: {e}")
    print("请检查是否在项目根目录运行、.env文件是否存在")
    exit(1)

print("\n" + "=" * 60)
print("第2步：检查文本有没有被正确填进Prompt模板")
print("=" * 60)

test_text = "宁德时代与特斯拉签署长期供货协议，订单量大幅增长。"

try:
    from chains.prompts.analysis_prompt import analysis_prompt
    prompt_str = analysis_prompt.format(text=test_text)
    is_ok = test_text in prompt_str
    print(f"测试文本: {test_text}")
    print(f"文本是否被正确替换进Prompt: {'✅ 是' if is_ok else '❌ 否，这就是问题所在！'}")
    if not is_ok:
        print("\n实际生成的Prompt内容如下（看看文本去哪了）：")
        print("-" * 40)
        print(prompt_str)
        print("-" * 40)
except Exception as e:
    print(f"❌ 第2步失败: {e}")
    exit(1)

print("\n" + "=" * 60)
print("第3步：直接调用LLM，看最原始的返回结果")
print("=" * 60)
print("（这一步会真实调用一次API，正常需要几秒到十几秒）\n")

try:
    from chains.qa_chain import get_llm
    from chains.prompts.analysis_prompt import AnalysisResult

    llm = get_llm().with_structured_output(AnalysisResult)
    response = llm.invoke(prompt_str)
    result = response.dict()

    print("LLM原始返回结果：")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print()
    print(f"公司名是否正确识别为'宁德时代': {'✅ 是' if '宁德时代' in str(result.get('company', '')) else '❌ 否'}")
    print(f"情感是否判断为'正面': {'✅ 是' if result.get('sentiment_label') == '正面' else '❌ 否'}")

except Exception as e:
    print(f"❌ 第3步调用LLM失败，报错信息如下：")
    print(f"   {type(e).__name__}: {e}")
    print("\n   这种情况说明问题出在LLM调用本身（比如API Key无效/网络/模型名称错误），")
    print("   不是Prompt或解析逻辑的问题。")
    exit(1)

print("\n" + "=" * 60)
print("第4步：跑完整的 analyze_text() 函数，和第3步对比")
print("=" * 60)

try:
    from chains.analysis_chain import analyze_text
    final_result = analyze_text(test_text)
    print("analyze_text() 最终返回结果：")
    for k, v in final_result.items():
        print(f"  {k}: {v}")

    if final_result.get("company") != result.get("company") or final_result.get("sentiment_label") != result.get("sentiment_label"):
        print("\n⚠️ 注意：第3步（原始LLM结果）和第4步（analyze_text最终结果）不一致！")
        print("   说明问题出在 _finalize_result 或缓存这一层，不是LLM本身的问题。")
    else:
        print("\n✅ 第3步和第4步结果一致，说明解析/缓存层没问题。")
except Exception as e:
    print(f"❌ 第4步失败: {e}")

print("\n" + "=" * 60)
print("诊断完成！请根据上方输出定位问题环节")
print("=" * 60)
