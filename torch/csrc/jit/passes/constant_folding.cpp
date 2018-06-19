#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit {

at::ArrayRef<torch::jit::Value *> runNode(Node *n) {
  auto graph = std::make_shared<Graph>();
  auto block = graph->block();
  
  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value * v) {
    auto it = local_map.find(v);
    if(it != local_map.end())
      return it->second;
    barf("Encountered a use of a value not in scope");
  };

  for(auto input : n->inputs()) {
    local_map[input] = block->addInput()->copyMetadata(input)->setStage(input->stage());
    graph->setStage(std::max(graph->stage(), input->stage()));
  }

  auto new_node = block->appendNode(graph->createClone(n, env, /*non-recursive */ false));
  new_node->setStage(n->stage());
  graph->setStage(std::max(graph->stage(), n->stage()));

  for(size_t i = 0; i < n->outputs().size(); ++i) {
    auto oo = n->outputs()[i];
    auto no = new_node->outputs()[i];
    local_map[oo] = no;
    no->copyMetadata(oo);
    no->setStage(oo->stage());
  }
  for(auto output : n->outputs()) {
    block->registerOutput(env(output));
  }
  auto values = fmap(block->inputs(), [&](Value* v) {
    return v->node()->t(attr::value);
  });
  InterpreterState(Code(graph)).runOneStage(values);
  return block->outputs();
}

void propagateNode(Node *n) {
  auto outputs = runNode(n);
  for(size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs()[i]->replaceAllUsesWith(outputs[i]);
  }
}

void ConstantFolding(Node* n, bool recurse) {
  auto & graph = *n->owningGraph();
  bool propagate = std::all_of(n->inputs.begin(), n->inputs.end(), [&](Value* v) {
    return v->node()->kind() == prim::Constant;
  }) && n->kind() != prim::Print;
  if (propagate) {
    propagateNode(n);
  }
  if (recurse) {
    for (Block * block : n->blocks())
      ConstantFolding(block, recurse);
  }
  if (propagate) {
    // n->destroy();
  }
}

void ConstantFolding(Block* block, bool recurse) {
  for(auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
    auto n = *it++;
    ConstantFolding(n, recurse);
  }
}

void ConstantFolding(std::shared_ptr<Graph>& graph) {
  ConstantFolding(graph->block(), true);
  EliminateDeadCode(graph);
}

}}
