#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit {

std::unordered_set<Symbol> skip_list = {
  //FIXME If & Loop require special casing because they cannot be run as a
  //single node.
  prim::If,
  prim::Loop,
  //FIXME Same problem as in DCE - cpp & python PythonOp and CppOp should be
  //FIXME treated as having side effects but ONNX depends on them being removed
  prim::Print,
 };

std::vector<at::Tensor> runNode(Node *n) {
  auto graph = std::make_shared<Graph>();
  auto block = graph->block();
  //very similar to Block::cloneFrom, but here we are cloning just the node,
  //and not recursively
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
  auto values = fmap(n->inputs(), [&](Value* v) {
    return v->node()->t(attr::value);
  });
  InterpreterState(Code(graph)).runOneStage(values);
  //outputs are in values now
  return values;
}


void propagateNode(Node *n) {
  auto outputs = runNode(n);
  auto graph = n->owningGraph();
  for(size_t i = 0; i < outputs.size(); ++i) {
    auto new_node = graph->createConstant(outputs[i])->insertBefore(n);
    n->outputs()[i]->replaceAllUsesWith(new_node->output());
    //let dce elimination remove n
  }
}

void ConstantPropagation(Node* n, bool recurse) {
  bool constant_inputs = (n->inputs().size() > 0) &&
    std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
      return v->node()->kind() == prim::Constant;
    });
  bool supported_node = skip_list.count(n->kind()) == 0;
  if (constant_inputs && supported_node) {
    propagateNode(n);
  }
  if (recurse) {
    for (Block * block : n->blocks())
      ConstantPropagation(block, recurse);
  }
}

void ConstantPropagation(Block* block, bool recurse) {
  ConstantPropagation(block->param_node(), recurse);
  for (auto n: block->nodes()) {
    ConstantPropagation(n, recurse);
  }
}

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagation(graph->block(), true);
  EliminateDeadCode(graph);
}

}}
