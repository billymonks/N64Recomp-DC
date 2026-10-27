#include <algorithm>
#include <cassert>
#include <fstream>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "recompiler/generator.h"

static const char* fpr_sync_view(const N64Recomp::FunctionRegPlan::FprWeb& web);

struct BinaryOpFields { std::string func_string; std::string infix_string; };

static std::vector<BinaryOpFields> c_op_fields = []() {
    std::vector<BinaryOpFields> ret{};
    ret.resize(static_cast<size_t>(N64Recomp::BinaryOpType::COUNT));
    std::vector<char> ops_setup{};
    ops_setup.resize(static_cast<size_t>(N64Recomp::BinaryOpType::COUNT));

    auto setup_op = [&ret, &ops_setup](N64Recomp::BinaryOpType op_type, const std::string& func_string, const std::string& infix_string) {
        size_t index = static_cast<size_t>(op_type);
        // Prevent setting up an operation twice.
        assert(ops_setup[index] == false && "Operation already setup!");
        ops_setup[index] = true;
        ret[index] = { func_string, infix_string };
    };

    setup_op(N64Recomp::BinaryOpType::Add32,     "ADD32",  "");
    setup_op(N64Recomp::BinaryOpType::Sub32,     "SUB32",  "");
    setup_op(N64Recomp::BinaryOpType::Add64,     "",       "+");
    setup_op(N64Recomp::BinaryOpType::Sub64,     "",       "-");
    setup_op(N64Recomp::BinaryOpType::And64,     "",       "&");
    setup_op(N64Recomp::BinaryOpType::AddFloat,  "",       "+");
    setup_op(N64Recomp::BinaryOpType::AddDouble, "",       "+");
    setup_op(N64Recomp::BinaryOpType::SubFloat,  "",       "-");
    setup_op(N64Recomp::BinaryOpType::SubDouble, "",       "-");
    setup_op(N64Recomp::BinaryOpType::MulFloat,  "MUL_S",  "");
    setup_op(N64Recomp::BinaryOpType::MulDouble, "MUL_D",  "");
    setup_op(N64Recomp::BinaryOpType::DivFloat,  "DIV_S",  "");
    setup_op(N64Recomp::BinaryOpType::DivDouble, "DIV_D",  "");
    setup_op(N64Recomp::BinaryOpType::Or64,      "",       "|");
    setup_op(N64Recomp::BinaryOpType::Nor64,     "~",      "|");
    setup_op(N64Recomp::BinaryOpType::Xor64,     "",       "^");
    setup_op(N64Recomp::BinaryOpType::Sll32,     "S32",    "<<");
    setup_op(N64Recomp::BinaryOpType::Sll64,     "",       "<<");
    setup_op(N64Recomp::BinaryOpType::Srl32,     "S32",    ">>");
    setup_op(N64Recomp::BinaryOpType::Srl64,     "",       ">>");
    setup_op(N64Recomp::BinaryOpType::Sra32,     "S32",    ">>"); // Arithmetic aspect will be taken care of by unary op for first operand.
    setup_op(N64Recomp::BinaryOpType::Sra64,     "",       ">>"); // Arithmetic aspect will be taken care of by unary op for first operand.
    setup_op(N64Recomp::BinaryOpType::Equal,     "",       "==");
    setup_op(N64Recomp::BinaryOpType::EqualFloat,"",       "==");
    setup_op(N64Recomp::BinaryOpType::EqualDouble,"",      "==");
    setup_op(N64Recomp::BinaryOpType::NotEqual,  "",       "!=");
    setup_op(N64Recomp::BinaryOpType::Less,      "",       "<");
    setup_op(N64Recomp::BinaryOpType::LessFloat, "",       "<");
    setup_op(N64Recomp::BinaryOpType::LessDouble,"",       "<");
    setup_op(N64Recomp::BinaryOpType::LessEq,    "",       "<=");
    setup_op(N64Recomp::BinaryOpType::LessEqFloat,"",      "<=");
    setup_op(N64Recomp::BinaryOpType::LessEqDouble,"",     "<=");
    setup_op(N64Recomp::BinaryOpType::Greater,   "",       ">");
    setup_op(N64Recomp::BinaryOpType::GreaterEq, "",       ">=");
    setup_op(N64Recomp::BinaryOpType::LD,        "LD",     "");
    setup_op(N64Recomp::BinaryOpType::LW,        "MEM_W",  "");
    setup_op(N64Recomp::BinaryOpType::LWU,       "MEM_WU", "");
    setup_op(N64Recomp::BinaryOpType::LH,        "MEM_H",  "");
    setup_op(N64Recomp::BinaryOpType::LHU,       "MEM_HU", "");
    setup_op(N64Recomp::BinaryOpType::LB,        "MEM_B",  "");
    setup_op(N64Recomp::BinaryOpType::LBU,       "MEM_BU", "");
    setup_op(N64Recomp::BinaryOpType::LDL,       "do_ldl", "");
    setup_op(N64Recomp::BinaryOpType::LDR,       "do_ldr", "");
    setup_op(N64Recomp::BinaryOpType::LWL,       "do_lwl", "");
    setup_op(N64Recomp::BinaryOpType::LWR,       "do_lwr", "");
    setup_op(N64Recomp::BinaryOpType::True,      "", "");
    setup_op(N64Recomp::BinaryOpType::False,     "", "");

    // Ensure every operation has been setup.
    for (char is_set : ops_setup) {
        assert(is_set && "Operation has not been setup!");
    }

    return ret;
}();

std::string N64Recomp::CGenerator::gpr_to_string(int gpr_index) const {
    if (gpr_index == 0) {
        return "0";
    }
    if (plan != nullptr && plan->locals && (plan->cached & (1u << gpr_index))) {
        return fmt::format("r{}", gpr_index);
    }
    return fmt::format("ctx->r{}", gpr_index);
}

void N64Recomp::CGenerator::emit_writeback(GprMask regs) const {
    for (int reg = 1; reg < 32; reg++) {
        if (regs & (1u << reg)) {
            fmt::print(output_file, "ctx->r{0} = r{0}; ", reg);
        }
    }
    if (regs) {
        fmt::print(output_file, "\n    ");
    }
}

void N64Recomp::CGenerator::emit_reload(GprMask regs) const {
    for (int reg = 1; reg < 32; reg++) {
        if (regs & (1u << reg)) {
            fmt::print(output_file, "r{0} = ctx->r{0}; ", reg);
        }
    }
    if (regs) {
        fmt::print(output_file, "\n    ");
    }
}

void N64Recomp::CGenerator::emit_function_prologue() const {
    if (plan == nullptr) {
        return;
    }
    if (plan->locals) {
        bool first = true;
        for (int reg = 1; reg < 32; reg++) {
            if (!(plan->declared & (1u << reg))) {
                continue;
            }
            fmt::print(output_file, "{}r{}", first ? "    gpr " : ", ", reg);
            if (plan->entry_load & (1u << reg)) {
                fmt::print(output_file, " = ctx->r{}", reg);
            }
            first = false;
        }
        if (!first) {
            fmt::print(output_file, ";\n");
        }
        first = true;
        for (int reg = 16; reg < 32; reg++) {
            if (plan->save_locals & (1u << reg)) {
                fmt::print(output_file, "{}save_r{}", first ? "    gpr " : ", ", reg);
                first = false;
            }
        }
        if (!first) {
            fmt::print(output_file, ";\n");
        }
        for (size_t w = 0; w < plan->fpr_webs.size(); w++) {
            const auto& web = plan->fpr_webs[w];
            const char* type = web.is64 ? "fpr" : "fpr32";
            if (std::find(plan->fpr_entry.begin(), plan->fpr_entry.end(), (int)w) != plan->fpr_entry.end()) {
                fmt::print(output_file, "    {0} f{1}_{2} = {{ .{3} = ctx->f{1}.{3} }};\n", type, web.reg, w, fpr_sync_view(web));
            }
            else {
                fmt::print(output_file, "    {} f{}_{};\n", type, web.reg, w);
            }
        }
    }
    else if (plan->poison && plan->poison_entry) {
        fmt::print(output_file, "    RECOMP_POISON(ctx, 0x{:08X}u, 0xBEEF0000u);\n", plan->poison_entry);
    }
    if (plan->trace && plan->poison_entry) {
        fmt::print(output_file, "    TRACE_SET(ctx, 0x{:08X}u, 0x{:08X}u);\n", plan->poison_entry, plan->vram | 1);
    }
}

void N64Recomp::CGenerator::emit_call_sync_before(size_t instr_index) const {
    if (plan == nullptr) {
        return;
    }
    auto it = plan->calls.find(instr_index);
    if (it == plan->calls.end()) {
        return;
    }
    if (plan->locals) {
        emit_writeback(it->second.writeback);
        emit_fpr_sync(it->second.fpr_writeback, true);
    }
    if (plan->trace && it->second.runtime) {
        fmt::print(output_file, "TRACE_SINK(ctx, 0xF0u, 0x{:08X}u, 'R');\n    ", plan->vram + (uint32_t)instr_index * 4);
    }
}

void N64Recomp::CGenerator::emit_call_sync_after(size_t instr_index) const {
    if (plan == nullptr) {
        return;
    }
    auto it = plan->calls.find(instr_index);
    if (it == plan->calls.end()) {
        return;
    }
    if (plan->locals) {
        emit_reload(it->second.reload);
        emit_fpr_sync(it->second.fpr_reload, false);
    }
    else if (plan->poison && it->second.poison) {
        fmt::print(output_file, "RECOMP_POISON(ctx, 0x{:08X}u, 0xDEAD0000u);\n    ", it->second.poison);
    }
    if (plan->trace) {
        if (it->second.runtime) {
            // Runtime code returns in v0 and v1.
            fmt::print(output_file, "ctx->taint &= ~0xCu; ");
        }
        if (it->second.poison) {
            fmt::print(output_file, "TRACE_SET(ctx, 0x{:08X}u, 0x{:08X}u);\n    ", it->second.poison,
                       plan->vram + (uint32_t)instr_index * 4);
        }
    }
}

void N64Recomp::CGenerator::emit_exit_sync(size_t instr_index) const {
    if (plan == nullptr || !plan->locals) {
        return;
    }
    auto it = plan->exits.find(instr_index);
    if (it != plan->exits.end()) {
        emit_writeback(it->second);
    }
    auto fit = plan->fpr_exits.find(instr_index);
    if (fit != plan->fpr_exits.end()) {
        emit_fpr_sync(fit->second, true);
    }
}

// An FPR operand through a union member: its web's local, or the context (odd registers are the high word of the
// even register below them).
std::string N64Recomp::CGenerator::fpr_string(const InstructionContext& ctx, int reg, const char* view) const {
    std::string local = fpr_web_local(ctx, reg);
    if ((reg & 1) && std::string_view{ view } == "u32l") {
        return local.empty() ? fmt::format("ctx->f_odd[({} - 1) * 2]", reg) : local + ".u32h";
    }
    return local.empty() ? fmt::format("ctx->f{}.{}", reg, view) : fmt::format("{}.{}", local, view);
}

// regcache = "gpr+fpr": the local of the FPR web an operand belongs to (its write if fpr_output, else its read),
// or "" when that web stays in the context.
int N64Recomp::CGenerator::fpr_web_index(const InstructionContext& ctx, int reg) const {
    if (plan == nullptr || !plan->locals || plan->fpr_webs.empty() || reg < 0 || reg >= 32) {
        return -1;
    }
    const auto& map = fpr_output ? plan->fpr_def_web : plan->fpr_use_web;
    auto it = map.find((uint32_t)ctx.instr_index * 32 + (uint32_t)reg);
    return it == map.end() ? -1 : it->second;
}

std::string N64Recomp::CGenerator::fpr_web_local(const InstructionContext& ctx, int reg) const {
    int w = fpr_web_index(ctx, reg);
    return w < 0 ? std::string{} : fmt::format("f{}_{}", reg & ~1, w);
}

// A single-precision web used in arithmetic: lwc1/swc1 move it with float loads and stores.
bool N64Recomp::CGenerator::fpr_float_web(const InstructionContext& ctx, int reg) const {
    int w = fpr_web_index(ctx, reg);
    return w >= 0 && !plan->fpr_webs[w].is64 && plan->fpr_webs[w].arith;
}

// The union member a web's local is synced and loaded through: float or double if it takes part in arithmetic,
// otherwise the raw word or doubleword.
static const char* fpr_sync_view(const N64Recomp::FunctionRegPlan::FprWeb& web) {
    return web.is64 ? (web.arith && !web.halves ? "d" : "u64") : (web.arith ? "fl" : "u32l");
}

void N64Recomp::CGenerator::emit_fpr_sync(const std::vector<int>& webs, bool writeback) const {
    for (int entry : webs) {
        int w = entry & ~FunctionRegPlan::FPR_HIGH_HALF;
        const auto& web = plan->fpr_webs[w];
        const char* view = (entry & FunctionRegPlan::FPR_HIGH_HALF) ? "u32h" : fpr_sync_view(web);
        if (writeback) {
            fmt::print(output_file, "ctx->f{0}.{2} = f{0}_{1}.{2}; ", web.reg, w, view);
        }
        else {
            fmt::print(output_file, "f{0}_{1}.{2} = ctx->f{0}.{2}; ", web.reg, w, view);
        }
    }
    if (!webs.empty()) {
        fmt::print(output_file, "\n    ");
    }
}

static std::string unsigned_reloc(const N64Recomp::InstructionContext& context) {
    switch (context.reloc_type) {
        case N64Recomp::RelocType::R_MIPS_HI16:
            return fmt::format("{}RELOC_HI16({}, {:#X})",
                context.reloc_tag_as_reference ? "REF_" : "", context.reloc_section_index, context.reloc_target_section_offset);
        case N64Recomp::RelocType::R_MIPS_LO16:
            return fmt::format("{}RELOC_LO16({}, {:#X})",
                context.reloc_tag_as_reference ? "REF_" : "", context.reloc_section_index, context.reloc_target_section_offset);
        default:
            throw std::runtime_error(fmt::format("Unexpected reloc type {}\n", static_cast<int>(context.reloc_type)));
    }
}

static std::string signed_reloc(const N64Recomp::InstructionContext& context) {
    return "(int16_t)" + unsigned_reloc(context);
}

void N64Recomp::CGenerator::get_operand_string(Operand operand, UnaryOpType operation, const InstructionContext& context, std::string& operand_string) const {
    switch (operand) {
        case Operand::Rd:
            operand_string = gpr_to_string(context.rd);
            break;
        case Operand::Rs:
            operand_string = gpr_to_string(context.rs);
            break;
        case Operand::Rt:
            operand_string = gpr_to_string(context.rt);
            break;
        case Operand::Fd:
            operand_string = fpr_string(context, context.fd, "fl");
            break;
        case Operand::Fs:
            operand_string = fpr_string(context, context.fs, "fl");
            break;
        case Operand::Ft:
            operand_string = fpr_string(context, context.ft, "fl");
            break;
        case Operand::FdDouble:
            operand_string = fpr_string(context, context.fd, "d");
            break;
        case Operand::FsDouble:
            operand_string = fpr_string(context, context.fs, "d");
            break;
        case Operand::FtDouble:
            operand_string = fpr_string(context, context.ft, "d");
            break;
        case Operand::FdU32L:
            operand_string = fpr_string(context, context.fd, "u32l");
            break;
        case Operand::FsU32L:
            operand_string = fpr_string(context, context.fs, "u32l");
            break;
        case Operand::FtU32L:
            operand_string = fpr_string(context, context.ft, "u32l");
            break;
        case Operand::FdU32H:
            assert(false);
            break;
        case Operand::FsU32H:
            assert(false);
            break;
        case Operand::FtU32H:
            assert(false);
            break;
        case Operand::FdU64:
            operand_string = fpr_string(context, context.fd, "u64");
            break;
        case Operand::FsU64:
            operand_string = fpr_string(context, context.fs, "u64");
            break;
        case Operand::FtU64:
            operand_string = fpr_string(context, context.ft, "u64");
            break;
        case Operand::ImmU16:
            if (context.reloc_type != N64Recomp::RelocType::R_MIPS_NONE) {
                operand_string = unsigned_reloc(context);
            }
            else {
                operand_string = fmt::format("{:#X}", context.imm16);
            }
            break;
        case Operand::ImmS16:
            if (context.reloc_type != N64Recomp::RelocType::R_MIPS_NONE) {
                operand_string = signed_reloc(context);
            }
            else {
                operand_string = fmt::format("{:#X}", (int16_t)context.imm16);
            }
            break;
        case Operand::Sa:
            operand_string = std::to_string(context.sa);
            break;
        case Operand::Sa32:
            operand_string = fmt::format("({} + 32)", context.sa);
            break;
        case Operand::Cop1cs:
            operand_string = fmt::format("c1cs");
            break;
        case Operand::Hi:
            operand_string = "hi";
            break;
        case Operand::Lo:
            operand_string = "lo";
            break;
        case Operand::Zero:
            operand_string = "0";
            break;
    }
    switch (operation) {
        case UnaryOpType::None:
            break;
        case UnaryOpType::ToS32:
            operand_string = "S32(" + operand_string + ")"; 
            break;
        case UnaryOpType::ToU32:
            operand_string = "U32(" + operand_string + ")"; 
            break;
        case UnaryOpType::ToS64:
            operand_string = "SIGNED(" + operand_string + ")"; 
            break;
        case UnaryOpType::ToU64:
            // Nothing to do here, they're already U64
            break;
        case UnaryOpType::Lui:
            operand_string = "S32(" + operand_string + " << 16)"; 
            break;
        case UnaryOpType::Mask5:
            operand_string = "(" + operand_string + " & 31)";
            break;
        case UnaryOpType::Mask6:
            operand_string = "(" + operand_string + " & 63)";
            break;
        case UnaryOpType::ToInt32:
            operand_string = "(int32_t)" + operand_string; 
            break;
        case UnaryOpType::NegateFloat:
            operand_string = (codegen != nullptr && codegen->fprcache ? "FNEG_S(" : "NEG_S(") + operand_string + ")";
            break;
        case UnaryOpType::NegateDouble:
            operand_string = "-" + operand_string;
            break;
        case UnaryOpType::AbsFloat:
            operand_string = (codegen != nullptr && codegen->fprcache ? "FABS_S(" : "ABS_S(") + operand_string + ")";
            break;
        case UnaryOpType::AbsDouble:
            operand_string = "fabs(" + operand_string + ")";
            break;
        case UnaryOpType::SqrtFloat:
            operand_string = "sqrtf(" + operand_string + ")";
            break;
        case UnaryOpType::SqrtDouble:
            operand_string = "sqrt(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertSFromW:
            operand_string = "CVT_S_W(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertWFromS:
            operand_string = "CVT_W_S(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertDFromW:
            operand_string = "CVT_D_W(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertWFromD:
            operand_string = "CVT_W_D(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertDFromS:
            operand_string = "CVT_D_S(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertSFromD:
            operand_string = "CVT_S_D(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertDFromL:
            operand_string = "CVT_D_L(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertLFromD:
            operand_string = "CVT_L_D(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertSFromL:
            operand_string = "CVT_S_L(" + operand_string + ")";
            break;
        case UnaryOpType::ConvertLFromS:
            operand_string = "CVT_L_S(" + operand_string + ")";
            break;
        case UnaryOpType::TruncateWFromS:
            operand_string = "TRUNC_W_S(" + operand_string + ")";
            break;
        case UnaryOpType::TruncateWFromD:
            operand_string = "TRUNC_W_D(" + operand_string + ")";
            break;
        case UnaryOpType::TruncateLFromS:
            operand_string = "TRUNC_L_S(" + operand_string + ")";
            break;
        case UnaryOpType::TruncateLFromD:
            operand_string = "TRUNC_L_D(" + operand_string + ")";
            break;
        // TODO these four operations should use banker's rounding, but roundeven is C23 so it's unavailable here.
        case UnaryOpType::RoundWFromS:
            operand_string = "lroundf(" + operand_string + ")";
            break;
        case UnaryOpType::RoundWFromD:
            operand_string = "lround(" + operand_string + ")";
            break;
        case UnaryOpType::RoundLFromS:
            operand_string = "llroundf(" + operand_string + ")";
            break;
        case UnaryOpType::RoundLFromD:
            operand_string = "llround(" + operand_string + ")";
            break;
        case UnaryOpType::CeilWFromS:
            operand_string = "S32(ceilf(" + operand_string + "))";
            break;
        case UnaryOpType::CeilWFromD:
            operand_string = "S32(ceil(" + operand_string + "))";
            break;
        case UnaryOpType::CeilLFromS:
            operand_string = "S64(ceilf(" + operand_string + "))";
            break;
        case UnaryOpType::CeilLFromD:
            operand_string = "S64(ceil(" + operand_string + "))";
            break;
        case UnaryOpType::FloorWFromS:
            operand_string = "S32(floorf(" + operand_string + "))";
            break;
        case UnaryOpType::FloorWFromD:
            operand_string = "S32(floor(" + operand_string + "))";
            break;
        case UnaryOpType::FloorLFromS:
            operand_string = "S64(floorf(" + operand_string + "))";
            break;
        case UnaryOpType::FloorLFromD:
            operand_string = "S64(floor(" + operand_string + "))";
            break;
    }
}

void N64Recomp::CGenerator::get_notation(BinaryOpType op_type, std::string& func_string, std::string& infix_string) const {
    func_string = c_op_fields[static_cast<size_t>(op_type)].func_string;
    infix_string = c_op_fields[static_cast<size_t>(op_type)].infix_string;
}

void N64Recomp::CGenerator::get_binary_expr_string(BinaryOpType type, const BinaryOperands& operands, const InstructionContext& ctx, const std::string& output, std::string& expr_string) const {
    thread_local std::string input_a{};
    thread_local std::string input_b{};
    thread_local std::string func_string{};
    thread_local std::string infix_string{};
    get_operand_string(operands.operands[0], operands.operand_operations[0], ctx, input_a);
    get_operand_string(operands.operands[1], operands.operand_operations[1], ctx, input_b);
    get_notation(type, func_string, infix_string);
    
    // These cases aren't strictly necessary and are just here for parity with the old recompiler output.
    if (type == BinaryOpType::Less && !((operands.operands[1] == Operand::Zero && operands.operand_operations[1] == UnaryOpType::None) || (operands.operands[0] == Operand::Fs || operands.operands[0] == Operand::FsDouble))) {
        expr_string = fmt::format("{} {} {} ? 1 : 0", input_a, infix_string, input_b);
    }
    else if (type == BinaryOpType::Equal && operands.operands[1] == Operand::Zero && operands.operand_operations[1] == UnaryOpType::None) {
        expr_string = "!" + input_a;
    }
    else if (type == BinaryOpType::NotEqual && operands.operands[1] == Operand::Zero && operands.operand_operations[1] == UnaryOpType::None) {
        expr_string = input_a;
    }
    // End unnecessary cases.

    // TODO encode these ops to avoid needing special handling.
    else if (type == BinaryOpType::LWL || type == BinaryOpType::LWR || type == BinaryOpType::LDL || type == BinaryOpType::LDR) {
        expr_string = fmt::format("{}(rdram, {}, {}, {})", func_string, output, input_a, input_b);
    }
    else if (!func_string.empty() && !infix_string.empty()) {
        expr_string = fmt::format("{}({} {} {})", func_string, input_a, infix_string, input_b);
    }
    else if (codegen != nullptr && codegen->sp_subword_fold && operands.operands[0] == Operand::Base && ctx.rs == 29 &&
             (type == BinaryOpType::LH || type == BinaryOpType::LHU || type == BinaryOpType::LB || type == BinaryOpType::LBU)) {
        expr_string = fmt::format("{}_SP({}, {})", func_string, input_b, input_a);
    }
    else if (!func_string.empty()) {
        expr_string = fmt::format("{}({}, {})", func_string, input_a, input_b);
    }
    else if (!infix_string.empty()) {
        expr_string = fmt::format("{} {} {}", input_a, infix_string, input_b);
    }
    else {
        // Handle special cases
        if (type == BinaryOpType::True) {
            expr_string = "1";
        }
        else if (type == BinaryOpType::False) {
            expr_string = "0";
        }
        assert(false && "Binary operation must have either a function or infix!");
    }
}

void N64Recomp::CGenerator::emit_function_start(const std::string& function_name, size_t func_index) const {
    (void)func_index;
    fmt::print(output_file,
        "{} void {}({}) {{\n"
        // these variables shouldn't need to be preserved across function boundaries, so make them local for more efficient output
        "    {}\n"
        "    int c1cs = 0;\n", // cop1 conditional signal
        plan != nullptr && plan->leaf_inline ? "RECOMP_LEAF_FUNC" : "RECOMP_FUNC",
        function_name,
        // The runtime compiles without strict aliasing, so without restrict every store to rdram would make the
        // compiler reload and re-store the context's registers around it.
        gpr_32bit ? "uint8_t* restrict rdram, recomp_context* restrict ctx" : "uint8_t* rdram, recomp_context* ctx",
        gpr_32bit ? "gpr hi = 0, lo = 0; uint64_t result = 0;" : "uint64_t hi = 0, lo = 0, result = 0;");
    if (codegen != nullptr && codegen->rdram_base) {
        fmt::print(output_file,
            "    uint8_t* restrict const rb = rdram - 0x80000000u;\n"
            "    #undef RECOMP_RB\n"
            "    #define RECOMP_RB rb\n");
    }
}

void N64Recomp::CGenerator::emit_function_end() const {
    if (plan != nullptr && plan->locals && plan->end_writeback) {
        fmt::print(output_file, "    ");
        emit_writeback(plan->end_writeback);
    }
    if (plan != nullptr && plan->locals && !plan->fpr_end.empty()) {
        fmt::print(output_file, "    ");
        emit_fpr_sync(plan->fpr_end, true);
    }
    fmt::print(output_file, ";}}\n");
}

void N64Recomp::CGenerator::emit_function_call_lookup(uint32_t addr) const {
    fmt::print(output_file, "{}(0x{:08X})(rdram, ctx);\n", codegen != nullptr && codegen->lookup_cache ? "LOOKUP_FUNC_CACHED" : "LOOKUP_FUNC", addr);
}

void N64Recomp::CGenerator::emit_function_call_by_register(int reg) const {
    fmt::print(output_file, "{}({})(rdram, ctx);\n", codegen != nullptr && codegen->lookup_cache ? "LOOKUP_FUNC_CACHED" : "LOOKUP_FUNC", gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_function_call_reference_symbol(const Context& context, uint16_t section_index, size_t symbol_index, uint32_t target_section_offset) const {
    (void)target_section_offset;
    const N64Recomp::ReferenceSymbol& sym = context.get_reference_symbol(section_index, symbol_index);
    fmt::print(output_file, "{}(rdram, ctx);\n", sym.name);
}

void N64Recomp::CGenerator::emit_function_call(const Context& context, size_t function_index) const {
    fmt::print(output_file, "{}(rdram, ctx);\n", context.functions[function_index].name);
}

void N64Recomp::CGenerator::emit_named_function_call(const std::string& function_name) const {
    fmt::print(output_file, "{}(rdram, ctx);\n", function_name);
}

void N64Recomp::CGenerator::emit_goto(const std::string& target) const {
    fmt::print(output_file,
        "    goto {};\n", target);
}

void N64Recomp::CGenerator::emit_label(const std::string& label_name) const {
    fmt::print(output_file,
        "{}:\n", label_name);
}

void N64Recomp::CGenerator::emit_jtbl_addend_declaration(const JumpTable& jtbl, int reg) const {
    std::string jump_variable = fmt::format("jr_addend_{:08X}", jtbl.jr_vram);
    fmt::print(output_file, "gpr {} = {};\n", jump_variable, gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_branch_condition(const ConditionalBranchOp& op, const InstructionContext& ctx) const {
    // Thread local variables to prevent allocations when possible.
    // TODO these thread locals probably don't actually help right now, so figure out a better way to prevent allocations.
    thread_local std::string expr_string{};
    get_binary_expr_string(op.comparison, op.operands, ctx, "", expr_string);
    fmt::print(output_file, "if ({}) {{\n", expr_string);
}

void N64Recomp::CGenerator::emit_branch_close() const {
    fmt::print(output_file, "}}\n");
}

void N64Recomp::CGenerator::emit_branch_value(const ConditionalBranchOp& op, const InstructionContext& ctx) const {
    thread_local std::string expr_string{};
    get_binary_expr_string(op.comparison, op.operands, ctx, "", expr_string);
    fmt::print(output_file, "{{ int brc = {};\n", expr_string);
}

void N64Recomp::CGenerator::emit_branch_on_value(const std::string& target) const {
    fmt::print(output_file, "if (brc) goto {}; }}\n", target);
}

bool N64Recomp::CGenerator::emit_planned_save(size_t instr_index) const {
    if (plan == nullptr || !plan->locals) {
        return false;
    }
    auto it = plan->save_ops.find(instr_index);
    if (it == plan->save_ops.end()) {
        return false;
    }
    const auto& op = it->second;
    switch (op.form) {
        case 'c':
            // The context still holds the entry value: the restore makes the local equal to it again.
            if (op.restore) {
                fmt::print(output_file, "    {} = ctx->r{};\n", gpr_to_string(op.reg), op.reg);
            }
            break;
        case 'l':
            if (op.restore) {
                fmt::print(output_file, "    {} = save_r{};\n", gpr_to_string(op.reg), op.reg);
            }
            else {
                fmt::print(output_file, "    save_r{} = {};\n", op.reg, gpr_to_string(op.reg));
            }
            break;
        default:
            break;
    }
    return true;
}

void N64Recomp::CGenerator::emit_switch_close() const {
    fmt::print(output_file, "}}\n");
}

void N64Recomp::CGenerator::emit_switch(const Context& recompiler_context, const JumpTable& jtbl, int reg) const {
    (void)recompiler_context;
    (void)reg;
    // TODO generate code to subtract the jump table address from the register's value instead.
    // Once that's done, the addend temp can be deleted to simplify the generator interface.
    std::string jump_variable = fmt::format("jr_addend_{:08X}", jtbl.jr_vram);

    fmt::print(output_file, "switch ({} >> 2) {{\n", jump_variable);
}

void N64Recomp::CGenerator::emit_case(int case_index, const std::string& target_label) const {
    fmt::print(output_file, "case {}: goto {}; break;\n", case_index, target_label);
}

void N64Recomp::CGenerator::emit_switch_error(uint32_t instr_vram, uint32_t jtbl_vram) const {
    fmt::print(output_file, "default: switch_error(__func__, 0x{:08X}, 0x{:08X});\n", instr_vram, jtbl_vram);
}

void N64Recomp::CGenerator::emit_return(const Context& context, size_t func_index) const {
    if (context.trace_mode) {
        fmt::print(output_file, "TRACE_RETURN()\n    ");
    }
    const std::string& return_hook = context.functions[func_index].return_hook;
    if (!return_hook.empty()) {
        fmt::print(output_file, "{{\n    {}    return;\n    }}\n", return_hook);
        return;
    }
    fmt::print(output_file, "return;\n");
}

void N64Recomp::CGenerator::emit_check_fr(int fpr) const {
    fmt::print(output_file, "CHECK_FR(ctx, {});\n    ", fpr);
}

void N64Recomp::CGenerator::emit_check_nan(int fpr, bool is_double) const {
    fmt::print(output_file, "NAN_CHECK(ctx->f{}.{}); ", fpr, is_double ? "d" : "fl");
}

void N64Recomp::CGenerator::emit_cop0_status_read(int reg) const {
    fmt::print(output_file, "{} = cop0_status_read(ctx);\n", gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_cop0_status_write(int reg) const {
    fmt::print(output_file, "cop0_status_write(ctx, {});", gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_cop1_cs_read(int reg) const {
    fmt::print(output_file, "{} = get_cop1_cs();\n", gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_cop1_cs_write(int reg) const {
    fmt::print(output_file, "set_cop1_cs({});\n", gpr_to_string(reg));
}

void N64Recomp::CGenerator::emit_muldiv(InstrId instr_id, int reg1, int reg2) const {
    switch (instr_id) {
        case InstrId::cpu_mult:
            fmt::print(output_file, "result = S64(S32({})) * S64(S32({})); lo = S32(result >> 0); hi = S32(result >> 32);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_dmult:
            fmt::print(output_file, "DMULT(S64({}), S64({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_multu:
            fmt::print(output_file, "result = U64(U32({})) * U64(U32({})); lo = S32(result >> 0); hi = S32(result >> 32);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_dmultu:
            fmt::print(output_file, "DMULTU(U64({}), U64({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_div:
            if (gpr_32bit) {
                // 64-bit division is a libgcc call on 32-bit hosts, so the runtime provides a 32-bit DIV32 that handles
                // s32(0x80000000) / -1 and division by zero itself.
                fmt::print(output_file, "DIV32(S32({}), S32({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
                break;
            }
            // Cast to 64-bits before division to prevent artihmetic exception for s32(0x80000000) / -1
            fmt::print(output_file, "lo = S32(S64(S32({0})) / S64(S32({1}))); hi = S32(S64(S32({0})) % S64(S32({1})));\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_ddiv:
            fmt::print(output_file, "DDIV(S64({}), S64({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_divu:
            if (gpr_32bit) {
                fmt::print(output_file, "DIVU32(U32({}), U32({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
                break;
            }
            fmt::print(output_file, "lo = S32(U32({0}) / U32({1})); hi = S32(U32({0}) % U32({1}));\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        case InstrId::cpu_ddivu:
            fmt::print(output_file, "DDIVU(U64({}), U64({}), &lo, &hi);\n", gpr_to_string(reg1), gpr_to_string(reg2));
            break;
        default:
            assert(false);
            break;
    }
}

void N64Recomp::CGenerator::emit_syscall(uint32_t instr_vram) const {
    fmt::print(output_file, "recomp_syscall_handler(rdram, ctx, 0x{:08X});\n", instr_vram);
}

void N64Recomp::CGenerator::emit_do_break(uint32_t instr_vram) const {
    fmt::print(output_file, "do_break({});\n", instr_vram);
}

void N64Recomp::CGenerator::emit_pause_self() const {
    fmt::print(output_file, "pause_self(rdram);\n");
}

void N64Recomp::CGenerator::emit_trigger_event(uint32_t event_index) const {
    fmt::print(output_file, "recomp_trigger_event(rdram, ctx, base_event_index + {});\n", event_index);
}

void N64Recomp::CGenerator::emit_comment(const std::string& comment) const {
    fmt::print(output_file, "// {}\n", comment);
}

void N64Recomp::CGenerator::process_binary_op(const BinaryOp& op, const InstructionContext& ctx) const {
    // Thread local variables to prevent allocations when possible.
    // TODO these thread locals probably don't actually help right now, so figure out a better way to prevent allocations.
    thread_local std::string output{};
    thread_local std::string expression{};
    fpr_output = true;
    get_operand_string(op.output, UnaryOpType::None, ctx, output);
    fpr_output = false;
    // addiu $sp, $sp, -N: frame offsets are taken from the new sp.
    if (codegen != nullptr && codegen->rdram_base && op.type == BinaryOpType::Add32 && op.output == Operand::Rt &&
        op.operands.operands[0] == Operand::Rs && op.operands.operands[1] == Operand::ImmS16 &&
        ctx.rt == 29 && ctx.rs == 29 && ctx.reloc_type == RelocType::R_MIPS_NONE && (int16_t)ctx.imm16 < 0) {
        thread_local std::string sp{};
        get_operand_string(Operand::Rs, UnaryOpType::None, ctx, sp);
        fmt::print(output_file, "{} = SP_ADJ({}, {:#X});\n", output, sp, (int16_t)ctx.imm16);
        return;
    }
    // lwc1 into a single web used in arithmetic: a float load.
    fpr_output = true;
    bool float_load = op.type == BinaryOpType::LW && op.output == Operand::FtU32L && fpr_float_web(ctx, ctx.ft);
    std::string load_local = float_load ? fpr_web_local(ctx, ctx.ft) : std::string{};
    fpr_output = false;
    if (float_load) {
        thread_local std::string base{};
        thread_local std::string imm{};
        get_operand_string(Operand::Base, UnaryOpType::None, ctx, base);
        get_operand_string(Operand::ImmS16, UnaryOpType::None, ctx, imm);
        fmt::print(output_file, "{}.fl = MEM_F({}, {});\n", load_local, base, imm);
        return;
    }
    get_binary_expr_string(op.type, op.operands, ctx, output, expression);
    fmt::print(output_file, "{} = {};\n", output, expression);
}

void N64Recomp::CGenerator::process_unary_op(const UnaryOp& op, const InstructionContext& ctx) const {
    // Thread local variables to prevent allocations when possible.
    // TODO these thread locals probably don't actually help right now, so figure out a better way to prevent allocations.
    thread_local std::string output{};
    thread_local std::string input{};
    fpr_output = true;
    get_operand_string(op.output, UnaryOpType::None, ctx, output);
    fpr_output = false;
    get_operand_string(op.input, op.operation, ctx, input);
    fmt::print(output_file, "{} = {};\n", output, input);
}

void N64Recomp::CGenerator::process_store_op(const StoreOp& op, const InstructionContext& ctx) const {
    // Thread local variables to prevent allocations when possible.
    // TODO these thread locals probably don't actually help right now, so figure out a better way to prevent allocations.
    thread_local std::string base_str{};
    thread_local std::string imm_str{};
    thread_local std::string value_input{};
    get_operand_string(Operand::Base, UnaryOpType::None, ctx, base_str);
    get_operand_string(Operand::ImmS16, UnaryOpType::None, ctx, imm_str);
    get_operand_string(op.value_input, UnaryOpType::None, ctx, value_input);

    enum class StoreSyntax {
        Func,
        FuncWithRdram,
        Assignment,
    };

    StoreSyntax syntax;
    std::string func_text;

    switch (op.type) {
        case StoreOpType::SD:
            func_text = "SD";
            syntax = StoreSyntax::Func;
            break;
        case StoreOpType::SDL:
            func_text = "do_sdl";
            syntax = StoreSyntax::FuncWithRdram;
            break;
        case StoreOpType::SDR:
            func_text = "do_sdr";
            syntax = StoreSyntax::FuncWithRdram;
            break;
        case StoreOpType::SW:
            func_text = "MEM_W";
            syntax = StoreSyntax::Assignment;
            break;
        case StoreOpType::SWL:
            func_text = "do_swl";
            syntax = StoreSyntax::FuncWithRdram;
            break;
        case StoreOpType::SWR:
            func_text = "do_swr";
            syntax = StoreSyntax::FuncWithRdram;
            break;
        case StoreOpType::SH:
            func_text = "MEM_H";
            syntax = StoreSyntax::Assignment;
            break;
        case StoreOpType::SB:
            func_text = "MEM_B";
            syntax = StoreSyntax::Assignment;
            break;
        case StoreOpType::SDC1:
            func_text = "SD";
            syntax = StoreSyntax::Func;
            break;
        case StoreOpType::SWC1:
            func_text = "MEM_W";
            syntax = StoreSyntax::Assignment;
            // swc1 from a single web used in arithmetic: a float store.
            if (fpr_float_web(ctx, ctx.ft)) {
                func_text = "MEM_F";
                value_input = fpr_web_local(ctx, ctx.ft) + ".fl";
            }
            break;
        default:
            throw std::runtime_error("Unhandled store op");
    }

    if (codegen != nullptr && codegen->sp_subword_fold && ctx.rs == 29 && (op.type == StoreOpType::SH || op.type == StoreOpType::SB)) {
        func_text += "_SP";
    }

    switch (syntax) {
        case StoreSyntax::Func:
            fmt::print(output_file, "{}({}, {}, {});\n", func_text, value_input, imm_str, base_str);
            break;
        case StoreSyntax::FuncWithRdram:
            fmt::print(output_file, "{}(rdram, {}, {}, {});\n", func_text, imm_str, base_str, value_input);
            break;
        case StoreSyntax::Assignment:
            fmt::print(output_file, "{}({}, {}) = {};\n", func_text, imm_str, base_str, value_input);
            break;
    }
}
