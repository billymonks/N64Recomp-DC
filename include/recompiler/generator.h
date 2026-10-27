#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include "recompiler/context.h"
#include "operations.h"

namespace N64Recomp {
    struct InstructionContext {
        int rd;
        int rs;
        int rt;
        int sa;

        int fd;
        int fs;
        int ft;

        int cop1_cs;

        uint16_t imm16;
        // The instruction's index in its function (regcache = "gpr+fpr" names FPR webs by it).
        size_t instr_index = 0;

        bool reloc_tag_as_reference;
        RelocType reloc_type;
        uint32_t reloc_section_index;
        uint32_t reloc_target_section_offset;
    };

    class Generator {
    public:
        virtual void process_binary_op(const BinaryOp& op, const InstructionContext& ctx) const = 0;
        virtual void process_unary_op(const UnaryOp& op, const InstructionContext& ctx) const = 0;
        virtual void process_store_op(const StoreOp& op, const InstructionContext& ctx) const = 0;
        virtual void emit_function_start(const std::string& function_name, size_t func_index) const = 0;
        virtual void emit_function_end() const = 0;
        virtual void emit_function_call_lookup(uint32_t addr) const = 0;
        virtual void emit_function_call_by_register(int reg) const = 0;
        // target_section_offset can each be deduced from symbol_index if the full context is available,
        // but for live recompilation the reference symbol list is unavailable so it's still provided.
        virtual void emit_function_call_reference_symbol(const Context& context, uint16_t section_index, size_t symbol_index, uint32_t target_section_offset) const = 0;
        virtual void emit_function_call(const Context& context, size_t function_index) const = 0;
        virtual void emit_named_function_call(const std::string& function_name) const = 0;
        virtual void emit_goto(const std::string& target) const = 0;
        virtual void emit_label(const std::string& label_name) const = 0;
        virtual void emit_jtbl_addend_declaration(const JumpTable& jtbl, int reg) const = 0;
        virtual void emit_branch_condition(const ConditionalBranchOp& op, const InstructionContext& ctx) const = 0;
        virtual void emit_branch_close() const = 0;
        virtual void emit_switch(const Context& recompiler_context, const JumpTable& jtbl, int reg) const = 0;
        virtual void emit_case(int case_index, const std::string& target_label) const = 0;
        virtual void emit_switch_error(uint32_t instr_vram, uint32_t jtbl_vram) const = 0;
        virtual void emit_switch_close() const = 0;
        virtual void emit_return(const Context& context, size_t func_index) const = 0;
        virtual void emit_check_fr(int fpr) const = 0;
        virtual void emit_check_nan(int fpr, bool is_double) const = 0;
        virtual void emit_cop0_status_read(int reg) const = 0;
        virtual void emit_cop0_status_write(int reg) const = 0;
        virtual void emit_cop1_cs_read(int reg) const = 0;
        virtual void emit_cop1_cs_write(int reg) const = 0;
        virtual void emit_muldiv(InstrId instr_id, int reg1, int reg2) const = 0;
        virtual void emit_syscall(uint32_t instr_vram) const = 0;
        virtual void emit_do_break(uint32_t instr_vram) const = 0;
        virtual void emit_pause_self() const = 0;
        virtual void emit_trigger_event(uint32_t event_index) const = 0;
        virtual void emit_comment(const std::string& comment) const = 0;
        // Register caching (codegen.regcache). The defaults leave every register in the context.
        virtual void set_reg_plan(const FunctionRegPlan* plan) { (void)plan; }
        virtual void emit_function_prologue() const {}
        virtual void emit_call_sync_before(size_t instr_index) const { (void)instr_index; }
        virtual void emit_call_sync_after(size_t instr_index) const { (void)instr_index; }
        virtual void emit_exit_sync(size_t instr_index) const { (void)instr_index; }
        // codegen.delay_once: a conditional branch's condition into a temporary ahead of its delay slot, and the
        // jump on it after the slot.
        virtual void emit_branch_value(const ConditionalBranchOp& op, const InstructionContext& ctx) const { emit_branch_condition(op, ctx); }
        virtual void emit_branch_on_value(const std::string& target) const { emit_goto(target); emit_branch_close(); }
        // codegen.skip_saves: emits the instruction's stack save or restore in C (FunctionRegPlan::save_ops) and
        // returns true, or returns false for an instruction without one.
        virtual bool emit_planned_save(size_t instr_index) const { (void)instr_index; return false; }
    };

    class CGenerator final : Generator {
    public:
        CGenerator(std::ostream& output_file, bool gpr_32bit = false, const Context::CodegenOptions* codegen = nullptr)
            : output_file(output_file), gpr_32bit(gpr_32bit), codegen(codegen) {};
        void process_binary_op(const BinaryOp& op, const InstructionContext& ctx) const final;
        void process_unary_op(const UnaryOp& op, const InstructionContext& ctx) const final;
        void process_store_op(const StoreOp& op, const InstructionContext& ctx) const final;
        void emit_function_start(const std::string& function_name, size_t func_index) const final;
        void emit_function_end() const final;
        void emit_function_call_lookup(uint32_t addr) const final;
        void emit_function_call_by_register(int reg) const final;
        void emit_function_call_reference_symbol(const Context& context, uint16_t section_index, size_t symbol_index, uint32_t target_section_offset) const final;
        void emit_function_call(const Context& context, size_t function_index) const final;
        void emit_named_function_call(const std::string& function_name) const final;
        void emit_goto(const std::string& target) const final;
        void emit_label(const std::string& label_name) const final;
        void emit_jtbl_addend_declaration(const JumpTable& jtbl, int reg) const final;
        void emit_branch_condition(const ConditionalBranchOp& op, const InstructionContext& ctx) const final;
        void emit_branch_close() const final;
        void emit_switch(const Context& recompiler_context, const JumpTable& jtbl, int reg) const final;
        void emit_case(int case_index, const std::string& target_label) const final;
        void emit_switch_error(uint32_t instr_vram, uint32_t jtbl_vram) const final;
        void emit_switch_close() const final;
        void emit_return(const Context& context, size_t func_index) const final;
        void emit_check_fr(int fpr) const final;
        void emit_check_nan(int fpr, bool is_double) const final;
        void emit_cop0_status_read(int reg) const final;
        void emit_cop0_status_write(int reg) const final;
        void emit_cop1_cs_read(int reg) const final;
        void emit_cop1_cs_write(int reg) const final;
        void emit_muldiv(InstrId instr_id, int reg1, int reg2) const final;
        void emit_syscall(uint32_t instr_vram) const final;
        void emit_do_break(uint32_t instr_vram) const final;
        void emit_pause_self() const final;
        void emit_trigger_event(uint32_t event_index) const final;
        void emit_comment(const std::string& comment) const final;
        void set_reg_plan(const FunctionRegPlan* plan) final { this->plan = plan; }
        void emit_function_prologue() const final;
        void emit_call_sync_before(size_t instr_index) const final;
        void emit_call_sync_after(size_t instr_index) const final;
        void emit_exit_sync(size_t instr_index) const final;
        void emit_branch_value(const ConditionalBranchOp& op, const InstructionContext& ctx) const final;
        void emit_branch_on_value(const std::string& target) const final;
        bool emit_planned_save(size_t instr_index) const final;
    private:
        std::string gpr_to_string(int gpr_index) const;
        void emit_writeback(GprMask regs) const;
        void emit_reload(GprMask regs) const;
        // regcache = "gpr+fpr": FPR web locals to and from the context, and an operand's web local ("" for the
        // context); fpr_output is set while an instruction's output operand is formatted.
        void emit_fpr_sync(const std::vector<int>& webs, bool writeback) const;
        int fpr_web_index(const InstructionContext& ctx, int reg) const;
        std::string fpr_web_local(const InstructionContext& ctx, int reg) const;
        bool fpr_float_web(const InstructionContext& ctx, int reg) const;
        std::string fpr_string(const InstructionContext& ctx, int reg, const char* view) const;
        mutable bool fpr_output = false;
        void get_operand_string(Operand operand, UnaryOpType operation, const InstructionContext& context, std::string& operand_string) const;
        void get_binary_expr_string(BinaryOpType type, const BinaryOperands& operands, const InstructionContext& ctx, const std::string& output, std::string& expr_string) const;
        void get_notation(BinaryOpType op_type, std::string& func_string, std::string& infix_string) const;
        std::ostream& output_file;
        bool gpr_32bit;
        const Context::CodegenOptions* codegen;
        const FunctionRegPlan* plan = nullptr;
    };
}

#endif
